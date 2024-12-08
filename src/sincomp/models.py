# -*- coding: utf-8 -*-

"""
多种方言字音编码器模型.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import datetime
import tensorflow as tf


class EncoderBase(tf.Module):
    """
    方言编码器的基类，输入方言点和字，输出改字在该点的读音.

    模型预测方言读音分为3个步骤：
        1. encode: 把输入编码为输入向量
        2. transform: 输入向量变换为输出向量
        3. decode: 根据输出向量预测读音

    子类需实现函数 _transform(self, dialect_emb, input_emb)，其中 dialect_emb 为方言向量，
    input_emb 为输入向量。当 self.residual 为假时，_transform 返回变换后的输出变量，
    否则返回残差，该残差与 input_emb 相加得变换后的输出变量。
    """

    def __init__(
        self,
        dialect_nums,
        input_nums,
        output_nums,
        dialect_emb_size=20,
        input_emb_size=20,
        output_emb_size=20,
        output_bias=True,
        residual=False,
        l2=0,
        name='encoder'
    ):
        """
        Parameters:
            dialect_nums (array-like of int): 每个方言信息的取值数，如方言点数
            input_nums (array-like of int): 每个输入的取值数，如字数
            output_nums (array-like of int): 每个输出的取值数，如声韵调数
            dialect_emb_size (int): 方言向量长度
            input_emb_size (int): 输入向量长度
            output_emb_size (int): 输出向量长度
            output_bias (bool): 是否为输出添加偏置
            residual (bool): 为真时，子类 _transform 返回值为残差
            l2 (float): L2 正则化系数
            name (str): 生成的模型名字
        """

        super().__init__(name=name)

        self.dialect_nums = dialect_nums
        self.input_nums = tuple(input_nums)
        self.output_nums = tuple(output_nums)
        self.dialect_emb_size = dialect_emb_size
        self.input_emb_size = input_emb_size
        self.output_emb_size = output_emb_size
        self.residual = residual
        self.l2 = l2

        init = tf.random_normal_initializer()

        # 在向量表最后追加一项作为缺失值的向量，下同
        self.dialect_embs = [tf.Variable(
            init(shape=(n + 1, self.dialect_emb_size), dtype=tf.float32),
            name=f'dialect_emb{i}'
        ) for i, n in enumerate(self.dialect_nums)]

        self.input_embs = [tf.Variable(
            init(shape=(n + 1, self.input_emb_size), dtype=tf.float32),
            name=f'input_emb{i}'
        ) for i, n in enumerate(self.input_nums)]

        self.output_embs = [tf.Variable(
            init(shape=(n + 1, self.output_emb_size), dtype=tf.float32),
            name=f'output_emb{i}'
        ) for i, n in enumerate(self.output_nums)]

        if output_bias:
            self.output_biases = [tf.Variable(
                init(shape=(n + 1,), dtype=tf.float32),
                name=f'output_bias{i}'
            ) for i, n in enumerate(self.output_nums)]

    def encode_dialect(self, dialect):
        """
        把方言点编码成向量.

        Parameters:
            dialect (`tensorflow.Tensor`):
                方言张量，形状为 batch_size * 1，内容为整数编码

        Returns:
            input_emb (`tensorflow.Tensor`):
                编码的方言向量，形状为 batch_size * self.dialect_emb_size
        """

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        return tf.reduce_mean(tf.stack(
            [tf.nn.embedding_lookup(
                self.dialect_embs[i],
                tf.where(dialect[:, i] >= 0, dialect[:, i], self.dialect_nums[i])
            ) for i in range(len(self.dialect_embs))],
            axis=2
        ), axis=-1)

    def encode(self, inputs):
        """
        把输入编码成向量.

        Parameters:
            inputs (tensorflow.Tensor):
                输入张量，形状为 batch_size * len(self.input_nums)，内容为整数编码

        Returns:
            input_emb (tensorflow.Tensor):
                编码的输入向量，形状为 batch_size * self.input_emb_size
        """

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        return tf.reduce_mean(tf.stack(
            [tf.nn.embedding_lookup(
                self.input_embs[i],
                tf.where(inputs[:, i] >= 0, inputs[:, i], self.input_nums[i])
            ) for i in range(len(self.input_embs))],
            axis=2
        ), axis=-1)

    def transform(self, dialect_emb, input_emb):
        """
        把输入向量变换为输出向量.

        Parameters:
            dialect_emb (tensorflow.Tensor): 方言向量
            input_emb (tensorflow.Tensor): 输出向量

        Returns:
            output_emb (tensorflow.Tensor):
                输出向量，形状为 dialect_emb.shape[0] * self.output_emb_size
        """

        output_emb = self._transform(dialect_emb, input_emb)
        # self.residual 为真时，_transform 返回的是残差，需加上 input_emb 得输出向量
        return input_emb + output_emb if self.residual else output_emb

    def decode(self, output_emb):
        """
        根据输出向量预测输出的对数几率.

        Parameters:
            output_emb (tensorflow.Tensor): 由输入向量变换成的输出向量

        Returns:
            logits (list of tensorflow.Tensor):
                输出张量的数组，每个张量形状为 output_emb.shape[0] * output_embs[i].shape[0]，
                内容为对数几率
        """

        logits = [tf.matmul(output_emb, e, transpose_b=True) \
            for e in self.output_embs]
        if hasattr(self, 'output_biases'):
            logits = [tf.nn.bias_add(l, b) \
                for l, b in zip(logits, self.output_biases)]

        return logits

    def forward(self, dialect, inputs):
        """
        正向传播，根据方言编码和输入输出对数几率.

        Parameters:
            dialect, inputs: self.encode 的输入

        Returns:
            logits: self.decode 的输出
        """

        dialect_emb = self.encode_dialect(dialect)
        input_emb = self.encode(inputs)
        output_emb = self.transform(dialect_emb, input_emb)
        return self.decode(output_emb)

    @tf.function
    def predict(self, dialect, inputs):
        """
        根据方言编码和输入预测输出编码.

        Parameters:
            dialect (aray-like): 方言编码，形状为 batch_size，batch_size 为批大小
            inputs (array-like):
                输入张量，形状为 batch_size * len(inputs_nums)，数组内容为整数编码

        Returns:
            outputs (tensorflow.Tensor):
                输出张量，每个张量的形状为 batch_size，内容为输出编码
        """

        dialect = tf.convert_to_tensor(dialect)
        inputs = tf.convert_to_tensor(inputs)

        logits = self.forward(dialect, inputs)
        # 最后一项代表缺失值，预测时不输出
        return tf.stack(
            [tf.argmax(l[:, :l.shape[1] - 1], axis=1, output_type=tf.int32) \
                for l in logits],
            axis=1
        )

    @tf.function
    def predict_proba(self, dialect, inputs):
        """
        根据方言编码和输入预测输出的概率.

        Parameters:
            dialect (aray-like): 方言编码，形状为 batch_size，batch_size 为批大小
            inputs (array-like):
                输入张量，形状为 batch_size * len(inputs_nums)，数组内容为整数编码

        Returns:
            probs (list of tensorflow.Tensor): 输出张量的数组，每个张量的形状为
                batch_size * self.output_nums[i]，内容为输出的概率
        """

        dialect = tf.convert_to_tensor(dialect)
        inputs = tf.convert_to_tensor(inputs)

        logits = self.forward(dialect, inputs)
        # 最后一项代表缺失值，预测时不输出
        return [tf.nn.softmax(l[:, :l.shape[1] - 1]) for l in logits]

    @tf.function
    def loss(self, dialect, inputs, targets):
        """
        根据方言编码、输入和目标输出计算损失.

        Parameters:
            dialect (aray-like): 方言编码，形状为 batch_size，batch_size 为批大小
            inputs (array-like):
                输入张量，形状为 batch_size * len(inputs_nums)，数组内容为整数编码
            targets (array-like):
                目标输出张量的数组，每个张量的形状为批大小，内容为输出编码

        Returns:
            loss (tensorflow.Tensor): 每个样本的损失，形状为 batch_size
            acc (tensorflow.Tensor): 每个样本的预测是否等于目标，
                形状为 batch_size * len(self.output_nums)
        """

        dialect = tf.convert_to_tensor(dialect)
        inputs = tf.convert_to_tensor(inputs)
        targets = tf.convert_to_tensor(targets)

        logits = self.forward(dialect, inputs)

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        loss = tf.stack(
            [tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.where(
                    targets[:, i] >= 0,
                    targets[:, i],
                    self.output_nums[i]
                ),
                logits=l
            ) for i, l in enumerate(logits)],
            axis=1
        )

        pred = tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )
        acc = tf.cast(targets == pred, tf.float32)

        return loss, acc

    @tf.function
    def update(self, optimizer, dialect, inputs, targets, weights):
        """
        反向传播更新模型参数.

        Parameters:
            optimizer (tensorflow.optimizers.Optimizer): 用于更新的优化器
            dialect, inputs, targets: self.loss 的输入
            weights (array-like): 各目标输出的权重

        Returns:
            loss, acc: self.loss 的返回值
        """

        dialect = tf.convert_to_tensor(dialect)
        inputs = tf.convert_to_tensor(inputs)
        targets = tf.convert_to_tensor(targets)
        if weights is not None:
            weights = tf.convert_to_tensor(weights)

        with tf.GradientTape() as tape:
            loss, acc = self.loss(dialect, inputs, targets)
            loss = tf.reduce_mean(loss, axis=0)
            loss = tf.reduce_sum(loss if weights is None else loss * weights)
            if self.l2 > 0:
                loss += self.l2 * tf.reduce_sum(
                    [tf.nn.l2_loss(v) for v in self.trainable_variables]
                )

        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss, tf.reduce_mean(acc, axis=0)

    def train(self, optimizer, data, weights=None):
        """
        使用数据集训练模型.

        Parameters:
            optimizer (tensorflow.optimizers.Optimizer): 用于更新的优化器
            data (tensorflow.data.Dataset): 训练数据集
            weights (array-like): 各目标的权重

        Returns:
            loss, acc (tensorflow.Tensor): 模型在训练数据集上的损失及精确度
        """

        if weights is not None:
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        loss_stat = tf.keras.metrics.Mean(dtype=tf.float32)
        acc_stats = [tf.keras.metrics.Mean(dtype=tf.float32) \
            for _ in self.output_embs]

        for dialect, inputs, targets in data:
            loss, acc = self.update(optimizer, dialect, inputs, targets, weights)
            loss_stat.update_state(loss)
            for i, s in enumerate(acc_stats):
                s.update_state(acc[i])

        return (
            loss_stat.result(),
            tf.convert_to_tensor([s.result() for s in acc_stats])
        )

    def evaluate(self, data, weights=None):
        """
        使用测试数据集评估模型.

        Parameters:
            data (tensorflow.data.Dataset): 测试数据集
            weights (array-like): 各目标的权重

        Returns:
            loss, acc (tensorflow.Tensor): 模型在测试数据集上的损失及精确度
        """

        if weights is not None:
            weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        loss_stat = tf.keras.metrics.Mean(dtype=tf.float32)
        acc_stats = [tf.keras.metrics.Mean(dtype=tf.float32) \
            for _ in self.output_embs]

        for dialect, inputs, targets in data:
            loss, acc = self.loss(dialect, inputs, targets)

            # 目标数据中的缺失值不计入
            loss = tf.where(targets >= 0, loss, 0)
            if weights is not None:
                loss = tf.tensordot(loss, weights, [-1, 0])
            loss_stat.update_state(loss)

            for i, s in enumerate(acc_stats):
                mask = targets[:, i] >= 0
                s.update_state(
                    tf.where(mask, acc[:, i], 0),
                    sample_weight=tf.cast(mask, tf.float32)
                )

        return (
            loss_stat.result(),
            tf.convert_to_tensor([s.result() for s in acc_stats])
        )

    def fit(
        self,
        optimizer,
        train_data,
        validate_data=None,
        weights=None,
        epochs=20,
        batch_size=100,
        output_path=None
    ):
        """
        训练模型.

        Parameters:
            optimizer (tensorflow.optimizers.Optimizer): 用于训练的优化器
            train_data (tensorflow.data.Dataset): 训练数据集
            validate_data (tensorflow.data.Dataset): 测试数据集
            weights (array-like): 各目标的权重
            epochs (int): 训练轮次
            batch_size (int): 批大小
            output_path (str): 检查点及统计数据输出路径

        训练过程中的检查点及统计数据输出到 output_path，如果 output_path 已有数据，
        先从最近一次检查点恢复训练状态。
        """

        if output_path is None:
            output_path = os.path.join(
                self.name,
                f'{datetime.datetime.now():%Y%m%d%H%M}'
            )

        logging.info(
            f'train {self.name}, epochs = {epochs}, weights = {weights}, '
            f'batch size = {batch_size}, output path = {output_path}'
        )

        train_writer = tf.summary.create_file_writer(os.path.join(output_path, 'train'))
        if validate_data is not None:
            validate_writer = tf.summary.create_file_writer(os.path.join(output_path, 'validate'))

        manager = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=self, optimizer=optimizer),
            os.path.join(output_path, 'checkpoints'),
            None
        )

        # 如果目标路径已包含检查点，先从检查点恢复
        if manager.restore_or_initialize() is not None:
            logging.info(f'restored from checkpoint {manager.latest_checkpoint}')

        while manager.checkpoint.save_counter < epochs:
            epoch = manager.checkpoint.save_counter.numpy() + 1
            loss, acc = self.train(
                optimizer,
                train_data.batch(batch_size),
                weights
            )

            logging.info(
                f'epoch {epoch}/{epochs}: '
                f'training loss = {loss}, accuracy = {acc}'
            )

            with train_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
                for i in range(acc.shape[0]):
                    tf.summary.scalar(f'accuracy{i}', acc[i], step=epoch)

                lr = optimizer.learning_rate
                if isinstance(lr, tf.optimizers.schedules.LearningRateSchedule):
                    lr = lr(optimizer.iterations)
                tf.summary.scalar('learning rate', lr, step=epoch)

                for v in self.variables:
                    tf.summary.histogram(v.name, v, step=epoch)

            if validate_data is not None:
                loss, acc = self.evaluate(validate_data.batch(batch_size), weights)
                logging.info(
                    f'epoch {epoch}/{epochs}: '
                    f'validation loss = {loss}, accuracy = {acc}'
                )

                with validate_writer.as_default():
                    tf.summary.scalar('loss', loss, step=epoch)
                    for i in range(acc.shape[0]):
                        tf.summary.scalar(f'accuracy{i}', acc[i], step=epoch)

            manager.save()

class BilinearEncoder(EncoderBase):
    """
    线性编码器.

    输入向量经过以方言向量为参数的线性变换得到输出向量。
    """

    def __init__(self, *args, name='bilinear_encoder', **kwargs):
        super().__init__(*args, name=name, **kwargs)

        init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            init(
                shape=(
                    self.dialect_emb_size,
                    self.input_emb_size,
                    self.output_emb_size
                ),
                dtype=tf.float32
            ),
            name='weight'
        )

    def _transform(self, dialect_emb, input_emb):
        """
        把输入向量变换为输出向量.

        输出向量为输入向量的线性变换，该变换的参数由方言向量经模型参数线性变换而来。
        """

        return tf.reshape(tf.matmul(
            input_emb[:, None, :],
            tf.tensordot(dialect_emb, self.weight, axes=[[-1], [0]])
        ), (input_emb.shape[0], self.weight.shape[-1]))
