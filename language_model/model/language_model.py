import collections
import os.path

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.language_model_util import *

__all__ = ["TrainResult", "EvaluateResult", "InferResult", "EncodeResult", "LanguageModel"]

class TrainResult(collections.namedtuple("TrainResult",
    ("loss", "learning_rate", "global_step", "batch_size", "summary"))):
    pass

class EvaluateResult(collections.namedtuple("EvaluateResult",
    ("loss", "word_count", "batch_size"))):
    pass

class InferResult(collections.namedtuple("InferResult",
    ("logit", "sample_id", "sample_word", "batch_size", "summary"))):
    pass

class EncodeResult(collections.namedtuple("EncodeResult",
    ("encoder_output", "encoder_output_length", "batch_size"))):
    pass

class LanguageModel(object):
    """forward-only language model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 vocab_size,
                 vocab_index,
                 vocab_inverted_index,
                 mode="train",
                 scope="lm"):
        """initialize language model"""
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.logger = logger
            self.hyperparams = hyperparams
            
            self.data_pipeline = data_pipeline
            self.vocab_size = vocab_size
            self.vocab_index = vocab_index
            self.vocab_inverted_index = vocab_inverted_index
            self.mode = mode
            self.scope = scope
            
            self.num_gpus = self.hyperparams.device_num_gpus
            self.default_gpu_id = self.hyperparams.device_default_gpu_id
            self.logger.log_print("# {0} gpus are used with default gpu id set as {1}"
                .format(self.num_gpus, self.default_gpu_id))
            
            """get batch inputs from data pipeline"""
            text_input = self.data_pipeline.text_input
            text_input_length = self.data_pipeline.text_input_length
            self.batch_size = tf.size(text_input_length)
            
            if self.mode == "encode" or self.mode == "infer":
                self.input_data_placeholder = self.data_pipeline.text_data_placeholder
                self.batch_size_placeholder = self.data_pipeline.batch_size_placeholder
            
            """build graph for language model"""
            self.logger.log_print("# build graph for language model")
            if self.mode == "encode":
                (encoder_output, _, _, input_embedding,
                    embedding_placeholder) = self._build_encoding_graph(text_input, text_input_length)
                self.encoder_output = encoder_output
                self.encoder_output_length = text_input_length
                self.input_embedding = input_embedding
                self.embedding_placeholder = embedding_placeholder
            else:
                (logit, _, _, _, input_embedding,
                    embedding_placeholder) = self._build_graph(text_input, text_input_length)
                self.input_embedding = input_embedding
                self.embedding_placeholder = embedding_placeholder
            
            if self.mode == "infer":
                sample_id, sample_word = self._generate_prediction(logit)
                self.infer_logit = logit
                self.infer_sample_id = sample_id
                self.infer_sample_word = sample_word
                
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train" or self.mode == "eval":
                logit_length = self.data_pipeline.text_output_length
                self.word_count = tf.reduce_sum(logit_length)
                
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                label = self.data_pipeline.text_output
                loss = self._compute_loss(logit, label, logit_length)
                self.train_loss = loss
                self.eval_loss = loss
                
                """apply learning rate decay"""
                self.logger.log_print("# setup learning rate decay mechanism")
                self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                    initializer=tf.zeros_initializer, trainable=False)
                self.learning_rate = tf.get_variable("learning_rate", dtype=tf.float32,
                    initializer=tf.constant(self.hyperparams.train_optimizer_learning_rate), trainable=False)
                decayed_learning_rate = self._apply_learning_rate_decay(self.learning_rate)
                
                """initialize optimizer"""
                self.logger.log_print("# initialize optimizer")
                self.optimizer = self._initialize_optimizer(decayed_learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                """create summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            self.ckpt_dir = self.hyperparams.train_ckpt_output_dir
            self.ckpt_name = os.path.join(self.ckpt_dir, "model_ckpt")
            self.ckpt_saver = tf.train.Saver()
    
    def _build_embedding(self,
                         input_data):
        """build embedding layer for language model"""
        embed_dim = self.hyperparams.model_embed_dim
        pretrained_embedding = self.hyperparams.model_pretrained_embedding
        
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create embedding for language model")
            embedding, embedding_placeholder = create_embedding(self.vocab_size,
                embed_dim, pretrained_embedding)
            input_embedding = tf.nn.embedding_lookup(embedding, input_data)
            
            return input_embedding, embedding_placeholder
    
    def _build_rnn_layer(self,
                         layer_input,
                         layer_input_length,
                         layer_id,
                         layer_direction):
        """build rnn layer for language model"""
        unit_dim = self.hyperparams.model_encoder_unit_dim
        unit_type = self.hyperparams.model_encoder_unit_type
        hidden_activation = self.hyperparams.model_encoder_hidden_activation
        forget_bias = self.hyperparams.model_encoder_forget_bias
        residual_connect = self.hyperparams.model_encoder_residual_connect
        drop_out = self.hyperparams.model_encoder_dropout
        device_spec = get_device_spec(self.default_gpu_id+layer_id, self.num_gpus)
        
        with tf.variable_scope("rnn/layer_{0}/{1}".format(layer_id, layer_direction), reuse=tf.AUTO_REUSE):
            cell = create_rnn_single_cell(unit_dim, unit_type, hidden_activation,
                forget_bias, residual_connect, drop_out, device_spec)
            layer_output, layer_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=layer_input,
                sequence_length=layer_input_length, dtype=tf.float32)
        
        return layer_output, layer_final_state
    
    def _build_encoder(self,
                       encoder_input,
                       encoder_input_length):
        """build encoder layer for language model"""
        num_layer = self.hyperparams.model_encoder_num_layer
        include_input = self.hyperparams.model_encoder_encoding_include_input
        
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# create hidden layer for encoder")
            layer_input = encoder_input
            layer_input_length = encoder_input_length
            
            encoder_layer_output = []
            encoder_layer_final_state = []
            if include_input == True:
                encoder_layer_output.append(encoder_input)
            
            for i in range(num_layer):
                layer_output, layer_final_state = self._build_rnn_layer(
                    layer_input, layer_input_length, i, "forward")
                encoder_layer_output.append(layer_output)
                encoder_layer_final_state.append(layer_final_state)
                layer_input = layer_output
            
            return encoder_layer_output, encoder_layer_final_state
    
    def _convert_encoder_output(self,
                                encoder_layer_output):
        """convert encoder output for language model"""
        encoding_type = self.hyperparams.model_encoder_encoding_type
        
        if encoding_type == "top":
            encoder_output = encoder_layer_output[-1]
        elif encoding_type == "bottom":
            encoder_output = encoder_layer_output[0]
        elif encoding_type == "average":
            encoder_output = tf.reduce_mean(encoder_layer_output, 0)
        else:
            raise ValueError("unsupported encoding type {0}".format(encoding_type))
        
        return encoder_output
    
    def _build_decoder(self,
                       decoder_input):
        """build decoder layer for language model"""
        projection_activation = self.hyperparams.model_decoder_projection_activation
        projection_activation_func = create_activation_function(projection_activation)
        
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            """create projection layer for decoder"""
            self.logger.log_print("# create projection layer for decoder")
            dense_projector = tf.layers.Dense(units=self.vocab_size, activation=projection_activation_func)
            decoder_output = dense_projector.apply(decoder_input)
            
            return decoder_output
    
    def _build_encoding_graph(self,
                              input_data,
                              input_length):
        """build encoding graph for language model"""
        self.logger.log_print("# build embedding layer for language model")
        input_embedding, embedding_placeholder = self._build_embedding(input_data)
        
        self.logger.log_print("# build encoder layer for language model")
        encoder_layer_output, encoder_layer_final_state = self._build_encoder(input_embedding, input_length)
        encoder_output = self._convert_encoder_output(encoder_layer_output)
        
        return (encoder_output, encoder_layer_output, encoder_layer_final_state,
            input_embedding, embedding_placeholder)
    
    def _build_graph(self,
                     input_data,
                     input_length):
        """build graph for language model"""       
        self.logger.log_print("# build embedding layer for language model")
        input_embedding, embedding_placeholder = self._build_embedding(input_data)
        
        self.logger.log_print("# build encoder layer for language model")
        encoder_layer_output, encoder_layer_final_state = self._build_encoder(input_embedding, input_length)
        encoder_output = self._convert_encoder_output(encoder_layer_output)
        
        self.logger.log_print("# build decoder layer for language model")
        decoder_output = self._build_decoder(encoder_output)
        
        return (decoder_output, encoder_output, encoder_layer_output,
            encoder_layer_final_state, input_embedding, embedding_placeholder)
    
    def _compute_loss(self,
                      logit,
                      label,
                      logit_length):
        """compute optimization loss"""
        mask = tf.sequence_mask(logit_length, maxlen=tf.shape(logit)[1], dtype=logit.dtype)
        cross_entropy = tf.contrib.seq2seq.sequence_loss(logits=logit, targets=label,
            weights=mask, average_across_timesteps=False, average_across_batch=True)
        loss = tf.reduce_sum(cross_entropy)
        
        return loss
    
    def _apply_learning_rate_decay(self,
                                   learning_rate):
        """apply learning rate decay"""
        decay_mode = self.hyperparams.train_optimizer_decay_mode
        decay_rate = self.hyperparams.train_optimizer_decay_rate
        decay_step = self.hyperparams.train_optimizer_decay_step
        decay_start_step = self.hyperparams.train_optimizer_decay_start_step
        
        if decay_mode == "exponential_decay":
            decayed_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        elif decay_mode == "inverse_time_decay":
            decayed_learning_rate = tf.train.inverse_time_decay(learning_rate=learning_rate,
                global_step=(self.global_step - decay_start_step), decay_steps=decay_step, decay_rate=decay_rate)
        else:
            raise ValueError("unsupported decay mode {0}".format(decay_mode))
        
        decayed_learning_rate = tf.cond(tf.less(self.global_step, decay_start_step),
            lambda: learning_rate, lambda: decayed_learning_rate)
        
        return decayed_learning_rate
    
    def _initialize_optimizer(self,
                              learning_rate):
        """initialize optimizer"""
        optimizer_type = self.hyperparams.train_optimizer_type
        if optimizer_type == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                momentum=self.hyperparams.train_optimizer_momentum_beta)
        elif optimizer_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                decay=self.hyperparams.train_optimizer_rmsprop_beta,
                epsilon=self.hyperparams.train_optimizer_rmsprop_epsilon)
        elif optimizer_type == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                rho=self.hyperparams.train_optimizer_adadelta_rho,
                epsilon=self.hyperparams.train_optimizer_adadelta_epsilon)
        elif optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,
                initial_accumulator_value=self.hyperparams.train_optimizer_adagrad_init_accumulator)
        elif optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                beta1=self.hyperparams.train_optimizer_adam_beta_1, beta2=self.hyperparams.train_optimizer_adam_beta_2,
                epsilon=self.hyperparams.train_optimizer_adam_epsilon)
        else:
            raise ValueError("unsupported optimizer type {0}".format(optimizer_type))
        
        return optimizer
    
    def _minimize_loss(self,
                       loss):
        """minimize optimization loss"""
        """compute gradients"""
        grads_and_vars = self.optimizer.compute_gradients(loss)
        
        """clip gradients"""
        gradients = [x[0] for x in grads_and_vars]
        variables = [x[1] for x in grads_and_vars]
        clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.hyperparams.train_clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        
        """update model based on gradients"""
        update_model = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        
        return update_model, clipped_gradients, gradient_norm
    
    def _generate_prediction(self,
                             logit):
        """generate prediction"""
        prediction_type = self.hyperparams.model_decoder_prediction_type
        
        if prediction_type == "max":
            sample_id = tf.argmax(logit, axis=-1)
        elif prediction_type == "sample":
            logit_shape = tf.shape(logit)
            batch_size, max_length, dim_size = logit_shape[0], logit_shape[1], logit_shape[2]
            logit_reshaped = tf.reshape(logit, shape=[-1, dim_size])
            logit_sampled = tf.multinomial(tf.log(logit_reshaped), num_samples=1)
            sample_id = tf.reshape(logit_sampled, shape=[batch_size, max_length])
        else:
            raise ValueError("unsupported prediction type {0}".format(prediction_type))
        
        sample_word = self.vocab_inverted_index.lookup(sample_id)
        
        return sample_id, sample_word
    
    def _get_train_summary(self):
        """get train summary"""
        return tf.summary.merge([tf.summary.scalar("learning_rate", self.learning_rate),
            tf.summary.scalar("train_loss", self.train_loss), tf.summary.scalar("gradient_norm", self.gradient_norm)])
    
    def train(self,
              sess,
              embedding):
        """train language model"""
        pretrained_embedding = self.hyperparams.model_pretrained_embedding
        
        if pretrained_embedding == True:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            _, loss, learning_rate, global_step, batch_size, summary = sess.run([self.update_model,
                self.train_loss, self.learning_rate, self.global_step, self.batch_size, self.train_summary])
        
        return TrainResult(loss=loss, learning_rate=learning_rate,
            global_step=global_step, batch_size=batch_size, summary=summary)
    
    def evaluate(self,
                 sess,
                 embedding):
        """evaluate language model"""
        pretrained_embedding = self.hyperparams.model_pretrained_embedding
        
        if pretrained_embedding == True:
            loss, word_count, batch_size = sess.run([self.eval_loss, self.word_count, self.batch_size],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            loss, word_count, batch_size = sess.run([self.eval_loss, self.word_count, self.batch_size])
        
        return EvaluateResult(loss=loss, word_count=word_count, batch_size=batch_size)
    
    def _get_infer_summary(self):
        """get infer summary"""
        return tf.no_op()
    
    def infer(self,
              sess,
              embedding):
        """infer language model"""
        pretrained_embedding = self.hyperparams.model_pretrained_embedding
        
        if pretrained_embedding == True:
            logit, sample_id, sample_word, batch_size, summary = sess.run([self.infer_logit,
                self.infer_sample_id, self.infer_sample_word, self.batch_size, self.infer_summary],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            logit, sample_id, sample_word, batch_size, summary = sess.run([self.infer_logit,
                self.infer_sample_id, self.infer_sample_word, self.batch_size, self.infer_summary])
        
        return InferResult(logit=logit, sample_id=sample_id,
            sample_word=sample_word, batch_size=batch_size, summary=summary)
    
    def encode(self,
               sess,
               embedding):
        """encode language model"""
        pretrained_embedding = self.hyperparams.model_pretrained_embedding
        
        if pretrained_embedding == True:
            encoder_output, encoder_output_length, batch_size = sess.run([self.encoder_output,
                self.encoder_output_length, self.batch_size],
                feed_dict={self.embedding_placeholder: embedding})
        else:
            encoder_output, encoder_output_length, batch_size = sess.run([self.encoder_output,
                self.encoder_output_length, self.batch_size])
        
        return EncodeResult(encoder_output=encoder_output,
            encoder_output_length=encoder_output_length, batch_size=batch_size)
    
    def save(self,
             sess,
             global_step):
        """save checkpoint for language model"""
        self.ckpt_saver.save(sess, self.ckpt_name, global_step=global_step)
    
    def restore(self,
                sess):
        """restore language model from checkpoint"""
        ckpt_file = tf.train.latest_checkpoint(self.ckpt_dir)
        if ckpt_file is not None:
            self.ckpt_saver.restore(sess, ckpt_file)
        else:
            raise FileNotFoundError("latest checkpoint file doesn't exist")