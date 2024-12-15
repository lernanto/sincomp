# -*- coding: utf-8 -*-

"""
多种方言字音编码器模型.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import platform
import torch
import torch._dynamo.config
import torch.utils.tensorboard


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


# torch.compile 在非 Linux 系统上的支持不是很好，先关闭
if platform.system() != 'Linux':
    torch._dynamo.config.disable = True


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
        missing_id: int = 0,
        output_bias: bool = True,
        residual: bool = False,
        dropout: float | None = None
    ):
        """
        Parameters:
            dialect_vocab_sizes: 每个方言信息的取值数，如方言点数
            char_vocab_sizes: 每个输入的取值数，如字数
            target_vocab_sizes: 每个输出的取值数，如声韵调数
            dialect_emb_size: 方言向量长度
            char_emb_size: 输入向量长度
            output_emb_size: 输出向量长度
            missing_id: 代表缺失值的 ID
            output_bias: 是否为输出添加偏置
            residual: 为真时，子类 _transform 返回值为残差
            dropout: 训练时输出向量的丢弃率
        """

        super().__init__()

        self.dialect_vocab_sizes = tuple(dialect_vocab_sizes)
        self.char_vocab_sizes = tuple(char_vocab_sizes)
        self.target_vocab_sizes = tuple(target_vocab_sizes)
        self.dialect_emb_size = dialect_emb_size
        self.char_emb_size = char_emb_size
        self.output_emb_size = output_emb_size
        self.missing_id = missing_id
        self.residual = residual

        # 在向量表最后追加一项作为缺失值的向量，下同
        self.dialect_embs = torch.nn.ModuleList(
            [torch.nn.Embedding(n, dialect_emb_size, padding_idx=missing_id) \
                for n in self.dialect_vocab_sizes]
        )

        self.char_embs = torch.nn.ModuleList(
            [torch.nn.Embedding(n, char_emb_size, padding_idx=missing_id) \
                for n in self.char_vocab_sizes]
        )

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(output_emb_size, n, bias=output_bias) \
                for n in self.target_vocab_sizes]
        )

        self.dropout = None if dropout is None else torch.nn.Dropout(dropout)

    def encode_dialect(self, dialects: torch.Tensor) -> torch.Tensor:
        """
        把方言点编码成向量.

        Parameters:
            dialects: 方言张量，形状为 batch_size * len(dialect_vocab_sizes)，内容为整数编码

        Returns:
            char_emb: 编码的方言向量，形状为 batch_size * self.dialect_emb_size
        """

        return torch.stack(
            [e(dialects[:, i]) for i, e in enumerate(self.dialect_embs)],
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

        return torch.stack(
            [e(chars[:, i]) for i, e in enumerate(self.char_embs)],
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

    @torch.compile
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

        if self.dropout is not None:
            output_emb = self.dropout(output_emb)

        return self.decode(output_emb)


class BilinearEncoder(EncoderBase):
    """
    双线性编码器.

    字向量经过以方言向量为参数的线性变换得到输出向量。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


torch.compile
def train(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: torch.utils.data.DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用数据集训练模型

    Parameters:
        model: 待训练模型
        loss: 损失函数
        optimizer: 用于训练的优化器
        data: 训练数据集

    Returns:
        loss, acc: 模型在训练数据集上的损失及准确率
    """

    old_mode = model.training
    model.train()
    count = 0
    total_loss = torch.zeros(())
    total_acc = torch.zeros(len(model.target_vocab_sizes))

    for (dialects, chars), targets in data:
        optimizer.zero_grad()
        logits = model(dialects, chars)
        lss = loss(logits, targets)
        (lss if lss.dim() == 0 else lss.mean()).backward()
        optimizer.step()

        preds = torch.stack([l.argmax(dim=-1) for l in logits], dim=1)
        count += dialects.size(0)
        total_loss += lss * dialects.size(0) if lss.dim() == 0 else lss.sum()
        total_acc += (preds == targets).to(total_acc.dtype).sum(dim=0)

    model.train(old_mode)
    return total_loss / count, total_acc / count

torch.compile
def evaluate(
    model: torch.nn.Module,
    loss: torch.nn.Module,
    data: torch.utils.data.DataLoader
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用测试数据集评估模型

    Parameters:
        model: 待评估模型
        loss: 损失函数
        data: 测试数据集

    Returns:
        loss, acc: 模型在测试数据集上的损失及准确率
    """

    old_mode = model.training
    model.eval()
    count = 0
    total_loss = torch.zeros(())
    total_acc = torch.zeros(len(model.target_vocab_sizes))

    with torch.no_grad():
        for (dialects, chars), targets in data:
            logits = model(dialects, chars)
            preds = torch.stack([l.argmax(dim=-1) for l in logits], dim=1)
            lss = loss(logits, targets)
            count += dialects.size(0)
            total_loss += lss * dialects.size(0) if lss.dim() == 0 else lss.sum()
            total_acc += (preds == targets).to(total_acc.dtype).sum(dim=0)

    model.train(old_mode)
    return total_loss / count, total_acc / count

def fit(
    model,
    train_data: torch.utils.data.Dataset,
    validate_data: torch.utils.data.Dataset | None = None,
    loss: torch.nn.Module = MultiTargetLoss(reduction='none'),
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    epochs: int = 20,
    batch_size: int = 100,
    num_workers: int = 0,
    checkpoint_dir: str | None = None,
    log_dir: str | None = None
) -> None:
    """
    训练模型

    Parameters:
        model: 待训练模型
        train_data: 训练数据集
        validate_data: 测试数据集
        loss: 损失函数
        optimizer: 用于训练的优化器
        lr_scheduler: 用于更新优化器学习率的调度器
        epochs: 训练轮次
        batch_size: 批大小
        num_workers: 加载数据的并行数
        checkpoint_dir: 检查点输出路径，如果该路径已有数据，先从最近一次检查点恢复训练状态
        log_dir: 统计数据输出路径
    """

    logger.debug(
        f'train model, epochs = {epochs}, batch_size = {batch_size}, '
        f'num_workers = {num_workers}, checkpoint_dir = {checkpoint_dir}, '
        f'log_dir = {log_dir} .'
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not isinstance(train_data, torch.utils.data.IterableDataset),
        num_workers=num_workers
    )
    if validate_data is not None:
        validate_data_loader = torch.utils.data.DataLoader(
            validate_data,
            batch_size=batch_size,
            num_workers=num_workers
        )

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters())

    if log_dir is not None:
        train_writer = torch.utils.tensorboard.SummaryWriter(
            os.path.join(log_dir, 'train')
        )
        if validate_data is not None:
            validate_writer = torch.utils.tensorboard.SummaryWriter(
                os.path.join(log_dir, 'validate')
            )

    epoch = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 如果目标路径已包含检查点，先从检查点恢复
        entries = [e for e in os.scandir(checkpoint_dir) if e.name.endswith('.pt')]
        if len(entries) > 0:
            entries.sort(
                key=lambda e: int(os.path.splitext(e.name)[0].split('-')[-1])
            )
            checkpoint = torch.load(entries[-1].path, weights_only=True)

            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    while epoch < epochs:
        epoch += 1

        lss, acc = train(model, loss, optimizer, train_data_loader)

        logger.info(
            f'epoch {epoch}/{epochs}: training loss = {lss}, '
            f'accuracy = {acc}'
        )

        if log_dir is not None:
            train_writer.add_scalar('loss', lss, epoch)
            for i, a in enumerate(acc):
                train_writer.add_scalar(f'accuracy{i}', a, epoch)

            train_writer.add_scalar(
                'learning rate',
                optimizer.param_groups[0]['lr'],
                epoch
            )

            for name, param in model.named_parameters():
                train_writer.add_histogram(name, param, epoch)

        if validate_data is not None:
            lss, acc = evaluate(model, loss, validate_data_loader)
            logger.info(
                f'epoch {epoch}/{epochs}: '
                f'validation loss = {lss}, accuracy = {acc}'
            )

            if log_dir is not None:
                validate_writer.add_scalar('loss', lss, epoch)
                for i, a in enumerate(acc):
                    validate_writer.add_scalar(f'accuracy{i}', a, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if checkpoint_dir is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            if lr_scheduler is not None:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'cpkt-{epoch}.pt')
            )

    if log_dir is not None:
        train_writer.close()
        if validate_data is not None:
            validate_writer.close()
