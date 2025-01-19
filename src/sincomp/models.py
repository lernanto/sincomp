# -*- coding: utf-8 -*-

"""
多种方言字音编码器模型
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import tensorflow as tf


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


class EncoderBase(tf.Module):
    """
    方言编码器的基类，输入方言点和字，输出改字在该点的读音

    模型预测方言读音分为3个步骤：
        1. encode: 把输入编码为输入向量
        2. transform: 输入向量变换为输出向量
        3. decode: 根据输出向量预测读音

    子类需实现函数 transform(self, dialect_emb, char_emb)，其中 dialect_emb 为方言向量，
    char_emb 为输入向量。
    """

    def __init__(
        self,
        dialect_vocab_sizes: list[int],
        char_vocab_sizes: list[int],
        target_vocab_sizes: list[int],
        dialect_emb_size: int = 20,
        char_emb_size: int = 20,
        output_emb_size: int = 20,
        missing_id: int | None = None,
        dropout: float | None = None,
        target_bias: bool = True
    ):
        """
        Parameters:
            dialect_vocab_sizes: 每个方言信息的取值数，如方言点数
            char_vocab_sizes: 每个输入的取值数，如字数
            target_vocab_sizes: 每个输出的取值数，如声韵调数
            dialect_emb_size: 方言向量长度
            char_emb_size: 输入向量长度
            output_emb_size: 输出向量长度
            missing_id: 代表缺失值的 ID，为空表示不接受缺失值
            dropout: 以一定的概率把数据向量的某些维度置 0，为空不丢弃任何维度
            target_bias: 是否为输出添加偏置
        """

        super().__init__()

        self.dialect_vocab_sizes = dialect_vocab_sizes
        self.char_vocab_sizes = tuple(char_vocab_sizes)
        self.target_vocab_sizes = tuple(target_vocab_sizes)
        self.dialect_emb_size = dialect_emb_size
        self.char_emb_size = char_emb_size
        self.output_emb_size = output_emb_size
        self.missing_id = missing_id
        self.dropout = dropout

        init = tf.random_normal_initializer()

        # 在向量表最后追加一项作为缺失值的向量，下同
        self.dialect_embs = [tf.Variable(
            init(shape=(n, self.dialect_emb_size), dtype=tf.float32),
            name=f'dialect_emb/{i}'
        ) for i, n in enumerate(self.dialect_vocab_sizes)]

        self.char_embs = [tf.Variable(
            init(shape=(n, self.char_emb_size), dtype=tf.float32),
            name=f'char_emb/{i}'
        ) for i, n in enumerate(self.char_vocab_sizes)]

        self.target_embs = [tf.Variable(
            init(shape=(n, self.output_emb_size), dtype=tf.float32),
            name=f'target_emb/{i}'
        ) for i, n in enumerate(self.target_vocab_sizes)]

        if target_bias:
            self.target_biases = [tf.Variable(
                init(shape=(n,), dtype=tf.float32),
                name=f'target_bias/{i}'
            ) for i, n in enumerate(self.target_vocab_sizes)]

    def encode_dialect(self, dialects: tf.Tensor) -> tf.Tensor:
        """
        把方言特征编码成向量

        Parameters:
            dialects: 方言特征，形状为 batch_size * len(dialect_vocab_sizes)，内容为整数编码

        Returns:
            char_emb: 编码的方言向量，形状为 batch_size * dialect_emb_size
        """

        return tf.reduce_mean(tf.stack(
            [tf.nn.embedding_lookup(self.dialect_embs[i], dialects[:, i]) \
                for i in range(len(self.dialect_embs))],
            axis=2
        ), axis=-1)

    def encode_char(self, chars: tf.Tensor) -> tf.Tensor:
        """
        把字特征编码成向量

        Parameters:
            chars: 字特征，形状为 batch_size * len(char_vocab_sizes)，内容为整数编码

        Returns:
            char_emb: 编码的输入向量，形状为 batch_size * char_emb_size
        """

        return tf.reduce_mean(tf.stack(
            [tf.nn.embedding_lookup(self.char_embs[i], chars[:, i]) \
                for i in range(len(self.char_embs))],
            axis=2
        ), axis=-1)

    def decode(self, output_emb: tf.Tensor) -> list[tf.Tensor]:
        """
        根据输出字向量预测目标的对数几率

        Parameters:
            output_emb: 由输入字向量变换成的输出字向量

        Returns:
            logits: 输出预测目标的对数几率的列表，列表长度为目标数量，
                每个元素形状为 batch_size * target_vocab_sizes[i]
        """

        logits = [tf.matmul(output_emb, e, transpose_b=True) \
            for e in self.target_embs]
        if hasattr(self, 'target_biases'):
            logits = [tf.nn.bias_add(l, b) \
                for l, b in zip(logits, self.target_biases)]

        return logits

    def __call__(
        self,
        dialects: tf.Tensor,
        chars: tf.Tensor,
        train: bool = False
    ) -> list[tf.Tensor]:
        """
        正向传播，根据方言和字特征输出目标的对数几率

        Parameters:
            dialects: 方言编码，形状为 batch_size * len(dialect_vocab_sizes)
            chars: 字编码，形状为 batch_size * len(char_vocab_sizes)
            train: 是否训练，训练时需应用 dropout，预测时不需要

        Returns:
            logits: self.decode 的输出
        """

        dialect_emb = self.encode_dialect(dialects)
        char_emb = self.encode_char(chars)

        output_emb = self.transform(dialect_emb, char_emb)
        if train and self.dropout is not None:
            output_emb = tf.nn.dropout(output_emb, self.dropout)

        return self.decode(output_emb)

    @tf.function
    def predict(self, dialects: tf.Tensor, chars: tf.Tensor) -> tf.Tensor:
        """
        根据方言和字特征预测目标编码

        Parameters:
            dialects: 方言编码，形状为 batch_size * len(dialect_vocab_sizes)
            chars: 字编码，形状为 batch_size * len(char_vocab_sizes)

        Returns:
            outputs: 目标编码，形状为 batch_size * len(target_vocab_sizes)
        """

        dialects = tf.convert_to_tensor(dialects)
        chars = tf.convert_to_tensor(chars)

        logits = self(dialects, chars)
        return tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )

    @tf.function
    def predict_proba(
        self,
        dialects: tf.Tensor,
        chars: tf.Tensor
    ) -> list[tf.Tensor]:
        """
        根据方言和字特征预测目标的概率

        Parameters:
            dialects: 方言编码，形状为 batch_size * len(dialect_vocab_sizes)
            chars: 字编码，形状为 batch_size * len(char_vocab_sizes)

        Returns:
            probs: 目标概率的列表，长度为目标数，
                每个元素的形状为 batch_size * target_vocab_sizes[i]
        """

        dialects = tf.convert_to_tensor(dialects)
        chars = tf.convert_to_tensor(chars)

        logits = self(dialects, chars)
        return [tf.nn.softmax(l) for l in logits]

    def loss(
        self,
        dialects: tf.Tensor,
        chars: tf.Tensor,
        targets: tf.Tensor,
        train: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        根据方言特征、字特征和目标编码计算损失

        Parameters:
            dialects: 方言编码，形状为 batch_size * len(dialect_vocab_sizes)
            chars: 字编码，形状为 batch_size * len(char_vocab_sizes)
            targets: 目标编码，形状问 batch_size * len(target_vocab_sizes)
            train: 是否训练，训练时需应用 dropout，预测时不需要

        Returns:
            loss: 每个样本的损失，形状为 batch_size
            acc: 每个样本的预测是否等于目标，形状为 batch_size * len(target_vocab_sizes)
        """

        logits = self(dialects, chars, train=train)

        loss = tf.stack(
            [tf.nn.sparse_softmax_cross_entropy_with_logits(
                targets[:, i],
                logits=l
            ) for i, l in enumerate(logits)],
            axis=1
        )

        pred = tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )
        acc = tf.cast(pred == targets, tf.float32)

        return loss, acc

    def update(
        self,
        optimizer: tf.optimizers.Optimizer,
        dialects: tf.Tensor,
        chars: tf.Tensor,
        targets: tf.Tensor,
        target_weights: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        反向传播更新模型参数

        Parameters:
            optimizer: 用于更新的优化器
            dialects, chars, targets: self.loss 的输入
            target_weights: 各目标输出的权重

        Returns:
            loss, acc: self.loss 的返回值
        """

        with tf.GradientTape() as tape:
            loss, acc = self.loss(dialects, chars, targets, train=True)
            if self.missing_id is not None:
                loss = tf.where(targets != self.missing_id, loss, 0)
            loss = tf.reduce_mean(
                tf.reduce_sum(loss, axis=-1) if target_weights is None \
                    else tf.tensordot(loss, target_weights, -1)
            )

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss, tf.reduce_mean(acc, axis=0)

    @tf.function
    def train(
        self,
        optimizer: tf.optimizers.Optimizer,
        data: tf.data.Dataset,
        target_weights: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        使用数据集训练模型

        Parameters:
            optimizer: 用于更新的优化器
            data: 训练数据集
            target_weights: 各目标的权重

        Returns:
            loss, acc: 模型在训练数据集上的损失及精确度
        """

        if target_weights is not None:
            target_weights = tf.convert_to_tensor(
                target_weights,
                dtype=tf.float32
            )

        count = tf.zeros((), dtype=tf.int32)
        total_loss = tf.zeros((), dtype=tf.float32)
        total_acc = tf.zeros(len(self.target_vocab_sizes), dtype=tf.float32)

        for (dialects, chars), targets in data:
            count += 1
            loss, acc = self.update(
                optimizer,
                dialects,
                chars,
                targets,
                target_weights
            )
            total_loss += loss
            total_acc += acc

        count = tf.cast(count, tf.float32)
        return total_loss / count, total_acc / count

    @tf.function
    def evaluate(
        self,
        data: tf.data.Dataset,
        target_weights: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        使用测试数据集评估模型

        Parameters:
            data: 测试数据集
            target_weights: 各目标的权重

        Returns:
            loss, acc: 模型在测试数据集上的损失及精确度
        """

        if target_weights is not None:
            target_weights = tf.convert_to_tensor(
                target_weights,
                dtype=tf.float32
            )

        loss_count = tf.zeros((), dtype=tf.int32)
        total_loss = tf.zeros((), dtype=tf.float32)
        acc_count = tf.zeros(len(self.target_vocab_sizes), dtype=tf.int32)
        total_acc = tf.zeros(len(self.target_vocab_sizes), dtype=tf.float32)

        for (dialects, chars), targets in data:
            loss, acc = self.loss(dialects, chars, targets)

            if self.missing_id is not None:
                # 目标数据中的缺失值不计入
                mask = targets != self.missing_id
                acc_count += tf.reduce_sum(tf.cast(mask, tf.int32), axis=0)
                total_acc += tf.reduce_sum(tf.where(mask, acc, 0), axis=0)
                loss = tf.where(mask, loss, 0)

            loss = tf.reduce_sum(loss, axis=-1) if target_weights is None \
                else tf.tensordot(loss, target_weights, -1)
            loss_count += tf.shape(targets)[0]
            total_loss += tf.reduce_sum(loss)

        return (
            total_loss / tf.cast(loss_count, tf.float32),
            total_acc/ tf.cast(acc_count, tf.float32)
        )

    def fit(
        self,
        optimizer: tf.optimizers.Optimizer,
        train_data: tf.data.Dataset,
        validate_data: tf.data.Dataset | None = None,
        target_weights: tf.data.Dataset | None = None,
        epochs: int = 20,
        batch_size: int = 100,
        checkpoint_dir: str | None = None,
        log_dir: str | None = None
    ) -> None:
        """
        训练模型

        Parameters:
            optimizer: 用于训练的优化器
            train_data: 训练数据集
            validate_data: 测试数据集
            target_weights: 各目标的权重
            epochs: 训练轮次
            batch_size: 批大小
            checkpoint_dir: 检查点输出目录
            log_dir: 统计数据输出目录

        训练过程中的检查点及统计数据输出到 checkpoint_dir，如果 checkpoint_dir 已有数据，
        先从最近一次检查点恢复训练状态。
        """

        logger.debug(
            f'train model, target_weights = {target_weights}, '
            f'epochs = {epochs}, batch_size = {batch_size}, '
            f'checkpoint_dir = {checkpoint_dir}, log_dir = {log_dir}.'
        )

        if log_dir is not None:
            writer = tf.summary.create_file_writer(log_dir)

        epoch = 0

        if checkpoint_dir is not None:
            manager = tf.train.CheckpointManager(
                tf.train.Checkpoint(model=self, optimizer=optimizer),
                checkpoint_dir,
                None
            )

            # 如果目标路径已包含检查点，先从检查点恢复
            if manager.restore_or_initialize() is not None:
                logger.info(f'restored from checkpoint {manager.latest_checkpoint}')
                epoch = manager.checkpoint.save_counter.numpy()

        while epoch < epochs:
            epoch += 1
            loss, acc = self.train(
                optimizer,
                train_data.batch(batch_size),
                target_weights
            )

            logger.info(
                f'epoch {epoch}/{epochs}: '
                f'training loss = {loss}, accuracy = {acc}'
            )

            if log_dir is not None:
                with writer.as_default():
                    tf.summary.scalar('loss/train', loss, step=epoch)
                    for i in range(acc.shape[0]):
                        tf.summary.scalar(
                            f'accuracy/train/{i}',
                            acc[i],
                            step=epoch
                        )

                    lr = optimizer.learning_rate
                    if isinstance(
                        lr,
                        tf.optimizers.schedules.LearningRateSchedule
                    ):
                        lr = lr(optimizer.iterations)
                    tf.summary.scalar('learning rate', lr, step=epoch)

                    for v in self.variables:
                        tf.summary.histogram(v.name, v, step=epoch)

            if validate_data is not None:
                loss, acc = self.evaluate(
                    validate_data.batch(batch_size),
                    target_weights
                )
                logger.info(
                    f'epoch {epoch}/{epochs}: '
                    f'validation loss = {loss}, accuracy = {acc}'
                )

                if log_dir is not None:
                    with writer.as_default():
                        tf.summary.scalar('loss/validation', loss, step=epoch)
                        for i in range(acc.shape[0]):
                            tf.summary.scalar(
                                f'accuracy/validation/{i}',
                                acc[i],
                                step=epoch
                            )

            if checkpoint_dir is not None:
                manager.save()


class BilinearEncoder(EncoderBase):
    """
    双线性编码器

    字向量经过以方言向量为参数的线性变换得到输出向量。
    """

    def __init__(self, *args, residual: bool = False, **kwargs):
        """
        Parameters:
            residual: 为真时，线性变换得到的是残差，需加上 char_emb 得到输出向量
        """

        super().__init__(*args, **kwargs)

        self.residual = residual

        init = tf.random_normal_initializer()
        self.weight = tf.Variable(
            init(
                shape=(
                    self.dialect_emb_size,
                    self.char_emb_size,
                    self.output_emb_size
                ),
                dtype=tf.float32
            ),
            name='weight'
        )

    def transform(self, dialect_emb: tf.Tensor, char_emb: tf.Tensor) -> tf.Tensor:
        """
        把输入向量变换为输出向量

        Parameters:
            dialect_emb: 方言向量
            char_emb: 输出向量

        Returns:
            output_emb: 输出向量，形状为 batch_size * output_emb_size

        输出向量为输入向量的线性变换，该变换的参数由方言向量经模型参数线性变换而来。
        """

        output_emb = tf.reshape(
            tf.matmul(
                char_emb[:, None, :],
                tf.tensordot(dialect_emb, self.weight, axes=[[-1], [0]])
            ),
            (-1, self.weight.shape[-1])
        )

        # self.residual 为真时，线性变换得到的是残差，需加上 char_emb 得输出向量
        return char_emb + output_emb if self.residual else output_emb