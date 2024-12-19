# -*- coding: utf-8 -*-

"""
多种方言字音编码器模型.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import datetime
import logging
import os
import torch
import torch.utils.tensorboard
import torchmetrics


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


class EncoderBase(torch.nn.Module):
    """
    方言编码器的基类，输入方言点和字，输出改字在该点的读音.

    模型预测方言读音分为3个步骤：
        1. encode: 把输入编码为输入向量
        2. transform: 输入向量变换为输出向量
        3. decode: 根据输出向量预测读音

    子类需实现函数 _transform(self, dialect_emb, char_emb)，其中 dialect_emb 为方言向量，
    char_emb 为输入向量。当 self.residual 为假时，_transform 返回变换后的输出变量，
    否则返回残差，该残差与 char_emb 相加得变换后的输出变量。
    """

    def __init__(
        self,
        dialect_vocab_sizes: list[int],
        char_vocab_sizes: list[int],
        target_vocab_sizes: list[int],
        dialect_emb_size: int = 20,
        char_emb_size: int = 20,
        output_emb_size: int = 20,
        output_bias: bool = True,
        residual: bool = False,
        l2: float = 0,
        name: str = 'encoder'
    ):
        """
        Parameters:
            dialect_vocab_sizes: 每个方言信息的取值数，如方言点数
            char_vocab_sizes: 每个输入的取值数，如字数
            target_vocab_sizes: 每个输出的取值数，如声韵调数
            dialect_emb_size: 方言向量长度
            char_emb_size: 输入向量长度
            output_emb_size: 输出向量长度
            output_bias: 是否为输出添加偏置
            residual: 为真时，子类 _transform 返回值为残差
            l2: L2 正则化系数
            name: 生成的模型名字
        """

        super().__init__()

        self.dialect_vocab_sizes = tuple(dialect_vocab_sizes)
        self.char_vocab_sizes = tuple(char_vocab_sizes)
        self.target_vocab_sizes = tuple(target_vocab_sizes)
        self.dialect_emb_size = dialect_emb_size
        self.char_emb_size = char_emb_size
        self.output_emb_size = output_emb_size
        self.residual = residual
        self.l2 = l2
        self.name = name

        # 在向量表最后追加一项作为缺失值的向量，下同
        self.dialect_embs = torch.nn.ModuleList(
            [torch.nn.Embedding(n + 1, dialect_emb_size) \
                for n in self.dialect_vocab_sizes]
        )

        self.char_embs = torch.nn.ModuleList(
            [torch.nn.Embedding(n + 1, char_emb_size) \
                for n in self.char_vocab_sizes]
        )

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(output_emb_size, n + 1, bias=output_bias) \
                for n in self.target_vocab_sizes]
        )

    def encode_dialect(self, dialects: torch.Tensor) -> torch.Tensor:
        """
        把方言点编码成向量.

        Parameters:
            dialects: 方言张量，形状为 batch_size * 1，内容为整数编码

        Returns:
            char_emb: 编码的方言向量，形状为 batch_size * self.dialect_emb_size
        """

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        return torch.stack(
            [self.dialect_embs[i](
                torch.where(
                    dialects[:, i] >= 0,
                    dialects[:, i],
                    self.dialect_vocab_sizes[i]
                )
            ) for i in range(len(self.dialect_embs))],
            dim=2
        ).mean(dim=-1)

    def encode_char(self, chars: torch.Tensor) -> torch.Tensor:
        """
        把输入编码成向量.

        Parameters:
            chars: 输入张量，形状为 batch_size * len(self.char_vocab_sizes)，内容为整数编码

        Returns:
            char_emb: 编码的输入向量，形状为 batch_size * self.char_emb_size
        """

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        return torch.stack(
            [self.char_embs[i](
                torch.where(chars[:, i] >= 0, chars[:, i], self.char_vocab_sizes[i])
            ) for i in range(len(self.char_embs))],
            dim=2
        ).mean(dim=-1)

    def transform(
        self,
        dialect_emb: torch.Tensor,
        char_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        把输入向量变换为输出向量.

        Parameters:
            dialect_emb: 方言向量
            char_emb: 输出向量

        Returns:
            output_emb: 输出向量，形状为 dialect_emb.size(0) * self.output_emb_size
        """

        output_emb = self._transform(dialect_emb, char_emb)
        # self.residual 为真时，_transform 返回的是残差，需加上 char_emb 得输出向量
        return char_emb + output_emb if self.residual else output_emb

    def decode(self, output_emb: torch.Tensor) -> list[torch.Tensor]:
        """
        根据输出向量预测输出的对数几率.

        Parameters:
            output_emb: 由输入向量变换成的输出向量

        Returns:
            logits: 输出张量的数组，每个张量形状为 output_emb.size(0) * linears[i].size(0)，
                内容为对数几率
        """

        return [l(output_emb) for l in self.linears]

    def forward(self, dialects: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """
        正向传播，根据方言编码和输入输出对数几率.

        Parameters:
            dialects, chars: self.encode 的输入

        Returns:
            logits: self.decode 的输出
        """

        dialect_emb = self.encode_dialect(dialects)
        char_emb = self.encode_char(chars)
        output_emb = self.transform(dialect_emb, char_emb)
        return self.decode(output_emb)

    def predict(self, dialects: list[int], chars: list[int]) -> torch.Tensor:
        """
        根据方言编码和输入预测输出编码.

        Parameters:
            dialects: 方言编码，形状为 batch_size，batch_size 为批大小
            chars: 输入张量，形状为 batch_size * len(char_vocab_sizes)，
                数组内容为整数编码

        Returns:
            outputs: 输出张量，每个张量的形状为 batch_size，内容为输出编码
        """

        dialects = torch.as_tensor(dialects)
        chars = torch.as_tensor(chars)

        logits = self.forward(dialects, chars)
        # 最后一项代表缺失值，预测时不输出
        return torch.stack(
            [l[:, :l.size(1) - 1].argmax(dim=1) for l in logits],
            dim=1
        )

    def predict_proba(
        self,
        dialects: list[int],
        chars: list[int]
    ) -> list[torch.Tensor]:
        """
        根据方言编码和输入预测输出的概率.

        Parameters:
            dialects: 方言编码，形状为 batch_size，batch_size 为批大小
            chars: 输入张量，形状为 batch_size * len(char_vocab_sizes)，数组内容为整数编码

        Returns:
            probs: 输出张量的数组，每个张量的形状为 batch_size * self.target_vocab_sizes[i]，
                内容为输出的概率
        """

        dialects = torch.as_tensor(dialects)
        chars = torch.as_tensor(chars)

        logits = self.forward(dialects, chars)
        # 最后一项代表缺失值，预测时不输出
        return [torch.nn.functional.softmax(l[:, :l.size(1) - 1], dim=-1) \
            for l in logits]

    def loss(
        self,
        dialects: list[int],
        chars: list[int],
        targets: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据方言编码、输入和目标输出计算损失.

        Parameters:
            dialects: 方言编码，形状为 batch_size，batch_size 为批大小
            chars: 输入张量，形状为 batch_size * len(char_vocab_sizes)，数组内容为整数编码
            targets: 目标输出张量的数组，每个张量的形状为批大小，内容为输出编码

        Returns:
            loss: 每个样本的损失，形状为 batch_size
            acc: 每个样本的预测是否等于目标，形状为 batch_size * len(self.target_vocab_sizes)
        """

        dialects = torch.as_tensor(dialects)
        chars = torch.as_tensor(chars)
        targets = torch.as_tensor(targets, dtype=torch.long)

        logits = self.forward(dialects, chars)

        # 输入中的 -1 代表缺失值，替换为最后一个向量
        loss = torch.stack(
            [torch.nn.functional.cross_entropy(
                l,
                torch.where(
                    targets[:, i] >= 0,
                    targets[:, i],
                    self.target_vocab_sizes[i]
                ),
                reduction='none'
            ) for i, l in enumerate(logits)],
            dim=1
        )

        pred = torch.stack([l.argmax(dim=1) for l in logits], axis=1)
        acc = (targets == pred).float()

        return loss, acc

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        dialects: list[int],
        chars: list[int],
        targets: list[int],
        weights: list[float] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        反向传播更新模型参数.

        Parameters:
            optimizer: 用于更新的优化器
            dialects, chars, targets: self.loss 的输入
            weights: 各目标输出的权重

        Returns:
            loss, acc: self.loss 的返回值
        """

        dialects = torch.as_tensor(dialects)
        chars = torch.as_tensor(chars)
        targets = torch.as_tensor(targets)
        if weights is not None:
            weights = torch.as_tensor(weights)

        optimizer.zero_grad()

        loss, acc = self.loss(dialects, chars, targets)
        loss = loss.mean(dim=0)
        loss = (loss if weights is None else loss * weights).sum()
        if self.l2 > 0:
            loss += self.l2 * torch.as_tensor(
                [torch.nn.MSELoss()(v) for v in self.trainable_variables]
            ).sum()

        loss.backward()
        optimizer.step()
        return loss, acc.mean(dim=0)

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        data: torch.utils.data.DataLoader,
        weights: list[float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用数据集训练模型.

        Parameters:
            optimizer: 用于更新的优化器
            data: 训练数据集
            weights: 各目标的权重

        Returns:
            loss, acc: 模型在训练数据集上的损失及精确度
        """

        if weights is not None:
            weights = torch.as_tensor(weights)

        loss_stat = torchmetrics.MeanMetric()
        acc_stats = [torchmetrics.MeanMetric() for _ in self.linears]

        for dialects, chars, targets in data:
            loss, acc = self.update(optimizer, dialects, chars, targets, weights)
            loss_stat.update(loss)
            for i, s in enumerate(acc_stats):
                s.update(acc[i])

        return (
            loss_stat.compute(),
            torch.as_tensor([s.compute() for s in acc_stats])
        )

    def evaluate(
        self,
        data: torch.utils.data.DataLoader,
        weights: list[float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        使用测试数据集评估模型.

        Parameters:
            data: 测试数据集
            weights: 各目标的权重

        Returns:
            loss, acc: 模型在测试数据集上的损失及精确度
        """

        if weights is not None:
            weights = torch.as_tensor(weights)

        loss_stat = torchmetrics.MeanMetric()
        acc_stats = [torchmetrics.MeanMetric() for _ in self.linears]

        for dialects, chars, targets in data:
            loss, acc = self.loss(dialects, chars, targets)

            # 目标数据中的缺失值不计入
            loss = torch.where(targets >= 0, loss, 0)
            if weights is not None:
                loss = torch.tensordot(loss, weights, ([-1], [0]))
            loss_stat.update(loss)

            for i, s in enumerate(acc_stats):
                mask = targets[:, i] >= 0
                s.update(
                    torch.where(mask, acc[:, i], 0),
                    weight=mask.float()
                )

        return (
            loss_stat.compute(),
            torch.as_tensor([s.compute() for s in acc_stats])
        )

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        train_data: torch.utils.data.Dataset,
        validate_data: torch.utils.data.Dataset | None = None,
        weights: list[float] | None = None,
        epochs: int = 20,
        batch_size: int = 100,
        output_path: str | None = None
    ) -> None:
        """
        训练模型.

        Parameters:
            optimizer: 用于训练的优化器
            train_data: 训练数据集
            validate_data: 测试数据集
            weights: 各目标的权重
            epochs: 训练轮次
            batch_size: 批大小
            output_path: 检查点及统计数据输出路径

        训练过程中的检查点及统计数据输出到 output_path，如果 output_path 已有数据，
        先从最近一次检查点恢复训练状态。
        """

        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        if validate_data is not None:
            validate_data_loader = torch.utils.data.DataLoader(
                validate_data,
                batch_size=batch_size
            )

        if output_path is None:
            output_path = os.path.join(
                self.name,
                f'{datetime.datetime.now():%Y%m%d%H%M}'
            )

        logger.info(
            f'train {self.name}, epochs = {epochs}, weights = {weights}, '
            f'batch size = {batch_size}, output path = {output_path}'
        )

        train_writer = torch.utils.tensorboard.SummaryWriter(
            os.path.join(output_path, 'train')
        )
        if validate_data is not None:
            validate_writer = torch.utils.tensorboard.SummaryWriter(
                os.path.join(output_path, 'validate')
            )

        checkpoint_dir = os.path.join(output_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        epoch = 0

        # 如果目标路径已包含检查点，先从检查点恢复
        entries = [e for e in os.scandir(checkpoint_dir) if e.is_file]
        if len(entries) > 0:
            entries.sort(key=lambda e: e.stat.st_mtime)
            latest = entries[-1].path
            checkpoint = torch.load(latest, weights_only=True)

            epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        while epoch < epochs:
            epoch += 1
            loss, acc = self.train(
                optimizer,
                train_data_loader,
                weights
            )

            logger.info(
                f'epoch {epoch}/{epochs}: '
                f'training loss = {loss}, accuracy = {acc}'
            )

            train_writer.add_scalar('loss', loss, epoch)
            for i in range(acc.size(0)):
                train_writer.add_scalar(f'accuracy{i}', acc[i], epoch)

            train_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

            for name, param in self.named_parameters():
                train_writer.add_histogram(name, param, epoch)

            if validate_data is not None:
                loss, acc = self.evaluate(validate_data_loader, weights)
                logger.info(
                    f'epoch {epoch}/{epochs}: '
                    f'validation loss = {loss}, accuracy = {acc}'
                )

                validate_writer.add_scalar('loss', loss, epoch)
                for i in range(acc.size(0)):
                    validate_writer.add_scalar(f'accuracy{i}', acc[i], epoch)

            path = os.path.join(checkpoint_dir, f'cpkt-{epoch}.pt')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                path
            )

class BilinearEncoder(EncoderBase):
    """
    双线性编码器.

    字向量经过以方言向量为参数的线性变换得到输出向量。
    """

    def __init__(self, *args, name: str = 'bilinear_encoder', **kwargs):
        super().__init__(*args, name=name, **kwargs)

        self.bilinear = torch.nn.Bilinear(
            self.dialect_emb_size,
            self.char_emb_size,
            self.output_emb_size,
            bias=False
        )

    def _transform(
        self,
        dialect_emb: torch.Tensor,
        char_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        把输入向量变换为输出向量.

        输出向量为输入向量的线性变换，该变换的参数由方言向量经模型参数线性变换而来。
        """

        return self.bilinear(dialect_emb, char_emb)


class MultiTargetLoss(torch.nn.Module):
    """
    计算方言读音声韵调等多目标的损失函数，为每个目标交叉熵损失之和
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -1,
        reduction: str = 'mean'
    ):
        """
        Parameters:
            weight: 每个目标的权重，为空时视为全 1
            ignore_index: 代表缺失值的 ID，不参与计算损失
            reduction: 把一批样本损失值规约成一个的方法
        """

        super().__init__()

        self.weight = None if weight is None \
            else torch.nn.Buffer(torch.as_tensor(weight))
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: list[torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多目标损失

        Parameters:
            logits: 每个目标的对数几率，形状为 batch_size * target_vocab_size
            targets: 目标标签，形状为 batch_size * target_num

        Returns:
            loss: 损失值
        """

        losses = torch.stack(
            [torch.nn.functional.cross_entropy(
                l,
                targets[:, i],
                ignore_index=self.ignore_index,
                reduction=self.reduction
            ) for i, l in enumerate(logits)],
            dim=-1
        )

        return losses.sum(dim=-1) if self.weight is None \
            else torch.tensordot(losses, self.weight, ([-1], [0]))