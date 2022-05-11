import argparse

import numpy as np
import torch
import tqdm
from numpy.random import dirichlet
from torch import optim
from torch.optim.lr_scheduler import StepLR

from . import utils, data_loader
from .logger import Logger
from .model import Encoder, Generator, Discriminator, GaussianGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tag', '--tag', type=str, default='default',
                        help='log directory of a training')

    parser.add_argument('-nh', '--hidden_size', type=int, default=100,
                        help='hidden size for the generator, the encoder and the discriminator')
    parser.add_argument('-nt', '--num_topics', type=int, default=20)
    parser.add_argument('-gs', '--gaussian', action='store_true',
                        help='whether to use GaussianGenerator')

    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.5,
                        help='Adam optimizer coefficient')
    parser.add_argument('-nc', '--num_d_steps', type=int, default=5,
                        help='number of discriminator optimization steps per training iteration')
    parser.add_argument('-alpha', '--alpha', type=float, default=.01,
                        help='Dirichlet parameter')
    parser.add_argument('-wc', '--weight_clipping', type=float, default=.01,
                        help='weight clipping')
    parser.add_argument('-dr', '--data_dir', type=str, default='data/20news_clean',
                        help='directory of the dataset to be used')

    parser.add_argument('-log', '--log_root', type=str, default='log',
                        help='root of all log directories')
    parser.add_argument('-ei', '--evaluation_interval', type=int, default=10,
                        help='evaluate the model every --evaluation_interval epochs')
    parser.add_argument('-nw', '--num_words', type=int, default=10,
                        help='save --num_words words for each topic on evaluation')
    parser.add_argument('-save', '--save', action='store_true',
                        help='whether to save model on evaluation')

    args = parser.parse_args()
    logger = Logger(args.log_root, args.tag, 'topic', 'ckpt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.seed_all(100)

    # load data
    vocabulary, data_iter, word_vectors = data_loader.load_data(args.data_dir, args.batch_size, args.gaussian)
    if args.gaussian:
        if word_vectors.shape[1] != args.hidden_size:
            raise ValueError(f'The dimension of pretrained word vectors is {word_vectors.shape[1]}, '
                             f'but got a different --hidden_size setting of {args.hidden_size}!')
        word_vectors = word_vectors.to(device)

    vocab_size = len(vocabulary)

    # build model
    if args.gaussian:
        gen = GaussianGenerator(args.num_topics, word_vectors, vocab_size).to(device)
    else:
        gen = Generator(args.num_topics, args.hidden_size, vocab_size).to(device)
    enc = Encoder(args.num_topics, args.hidden_size, vocab_size).to(device)
    dis = Discriminator(args.num_topics, args.hidden_size, vocab_size).to(device)

    # build optimizers
    betas = (args.adam_beta1, 0.999)
    d_opt = optim.Adam(dis.parameters(), lr=args.lr, betas=betas)
    if args.gaussian:
        g_opt = optim.Adam([
            {'params': gen.parameters(), 'lr': args.lr * 10},
            {'params': enc.parameters(), 'lr': args.lr},
        ], betas=betas)
    else:
        g_opt = optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=args.lr, betas=betas)

    d_scheduler = StepLR(d_opt, step_size=10, gamma=0.9)
    g_scheduler = StepLR(g_opt, step_size=10, gamma=0.9)

    logger.log(utils.training_stats(args, [gen, enc, dis],
                                    data_iter.dataset, data_iter, vocabulary))

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0
        total_step = 0
        for step, (enc_x,) in tqdm.tqdm(enumerate(data_iter), total=len(data_iter)):
            enc.train()
            gen.train()
            dis.train()

            batch_size = enc_x.shape[0]
            enc_x = enc_x.to(device)#真实样本数据
            gen_z = dirichlet(args.alpha * np.ones(args.num_topics), batch_size)#狄利克雷随机采样
            gen_z = torch.as_tensor(gen_z, dtype=torch.float32, device=device)

            gen_x = gen(gen_z)
            enc_z = enc(enc_x)

            # train the discriminator
            for i in range(args.num_d_steps):
                enc_prob = dis(torch.cat((enc_x, enc_z.detach()), dim=-1))#正样本对
                gen_prob = dis(torch.cat((gen_x.detach(), gen_z), dim=-1))#负样本对
                d_loss = (gen_prob - enc_prob).mean()#用梯度下降，最小化d_loss的过程，就是让
                #gen_prob变低，enc_prob增大,也就是让正样本对的得分越高，负样本对的得分越低

                #detach()的意思就是不参与训练

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                # weight clipping
                for p in dis.parameters():
                    with torch.no_grad():
                        p.clamp_(-args.weight_clipping, args.weight_clipping)

            # train the generator and the encoder
            enc_prob = dis(torch.cat((enc_x, enc_z), dim=-1))
            gen_prob = dis(torch.cat((gen_x, gen_z), dim=-1))
            g_loss = (gen_prob - enc_prob).mean()

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if isinstance(gen, GaussianGenerator):
                gen.clamp_cov_diag()

            total_loss += g_loss.item()
            total_step += 1

        d_scheduler.step()
        g_scheduler.step()

        logger.log(f'{utils.cur_hms()} | epoch {epoch:03d} | g_lr: {utils.get_lr(g_opt)} '
                   f'| d_lr: {utils.get_lr(d_opt)} | loss: {total_loss / total_step:.5f}')

        # evaluation
        if epoch % args.evaluation_interval == 0 or epoch == args.num_epochs or epoch == 1:
            enc.eval()
            gen.eval()
            dis.eval()
            topic_words, topics = get_topic_words(gen, args.num_words, vocabulary)
            logger.save_text(topic_words, f'{epoch:03d}.txt')
            logger.save_model({'t2d': gen, 'd2t': enc, 'dis': dis}, f'{epoch:03d}.ckpt.pt')


def get_topic_words(generator, num_words, vocabulary):
    with torch.no_grad():
        topics = generator.topic_word_dist
    topk_value, topk_idx = torch.topk(topics, k=num_words, dim=1, largest=True)
    words = [[vocabulary.id2word[y] for y in x] for x in topk_idx.tolist()]
    words = '\n'.join(' '.join(x) for x in words)
    return words, topics


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        main()
