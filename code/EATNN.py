import numpy as np
import tensorflow as tf
import os
import pandas as pd
import scipy.sparse
import time
import sys

DATA_ROOT = '../data/ciao'
f1=open(os.path.join(DATA_ROOT, 'EATNN.txt'), 'w')

unique_uid = list()
with open(os.path.join(DATA_ROOT, 'unique_uid_sub.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
unique_sid = list()
with open(os.path.join(DATA_ROOT, 'unique_sid_sub.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_songs = len(unique_sid)
n_users = len(unique_uid)


def load_data(csv_file):
    tp = pd.read_csv(csv_file)
    return tp


print n_songs
tp_test = load_data(os.path.join(DATA_ROOT, 'test.csv'))

tp_train = load_data(os.path.join(DATA_ROOT, 'train.csv'))

#tp_valid = load_data(os.path.join(DATA_ROOT, 'valid.csv'))

trust=load_data(os.path.join(DATA_ROOT, 'trust.csv'))

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class EATNN:
    def __init__(self, user_num, item_num, embedding_size,attention_size,max_item_pu,max_friend_pu):

        self.user_num=user_num
        self.item_num=item_num
        self.embedding_size=embedding_size
        self.attention_size=attention_size
        self.max_item_pu=max_item_pu
        self.max_friend_pu=max_friend_pu
        self.weight1=0.1
        self.weight2=0.1
        self.mu=0.1
        self.lambda_bilinear = [1e-3,1e-1, 1e-2]

    def _create_placeholders(self):

        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_i = tf.placeholder(tf.int32, [None, 1], name='input_iid')

        self.input_ur=tf.placeholder(tf.int32, [None, self.max_item_pu], name="input_ur")
        self.input_uf = tf.placeholder(tf.int32, [None, self.max_friend_pu], name="input_ur")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _create_variables(self):

        self.uidW_g = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.uidW_i = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                      stddev=0.01), dtype=tf.float32, name="uidWi")
        self.uidW_s = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                      stddev=0.01), dtype=tf.float32, name="uidWs")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")
        self.fidW = tf.Variable(tf.truncated_normal(shape=[self.user_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="fidW")

        # item domain
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")

        # social domain
        self.H_f = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hf")

        # item domain attention
        self.WA = tf.Variable(
            tf.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='WA')
        self.BA = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BA")
        self.HA = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HA")

        # social domain attention
        self.WB = tf.Variable(
            tf.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='WB')
        self.BB = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BB")
        self.HB = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HB")

    def _item_attentive_transfer(self):

        item_w=tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_i,self.WA)+self.BA),self.HA))
        general_w=tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_g,self.WA)+self.BA),self.HA))
        item_w=tf.div(item_w,item_w+general_w)
        general_w=1.0-item_w
        uid_A=item_w*self.uid_i+general_w*self.uid_g
        return uid_A,item_w

    def _social_attentive_transfer(self):

        social_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_s, self.WB) + self.BB), self.HB))
        general_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_g, self.WB) + self.BB), self.HB))
        social_w = tf.div(social_w, social_w + general_w)
        general_w = 1.0 - social_w
        uid_B = social_w * self.uid_s + general_w * self.uid_g
        return uid_B,social_w

    def _create_inference(self):
        self.uid_g = tf.nn.embedding_lookup(self.uidW_g, self.input_u)
        self.uid_g = tf.reshape(self.uid_g, [-1, self.embedding_size])
        self.uid_i = tf.nn.embedding_lookup(self.uidW_i, self.input_u)
        self.uid_i = tf.reshape(self.uid_i, [-1, self.embedding_size])
        self.uid_s = tf.nn.embedding_lookup(self.uidW_s, self.input_u)
        self.uid_s = tf.reshape(self.uid_s, [-1, self.embedding_size])

        self.uid_A,self.item_w=self._item_attentive_transfer()
        self.uid_B,self.social_w=self._social_attentive_transfer()

        #self.uid_A=tf.nn.relu(self.uid_A)
        #self.uid_B=tf.nn.relu(self.uid_B)

        self.uid_A = tf.nn.dropout(self.uid_A, self.dropout_keep_prob)
        self.uid_B = tf.nn.dropout(self.uid_B, self.dropout_keep_prob)

        self.pos_item=tf.nn.embedding_lookup(self.iidW,self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r=tf.einsum('ac,abc->abc',self.uid_A,self.pos_item)
        self.pos_r=tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, max_item_pu])

        self.pos_friend = tf.nn.embedding_lookup(self.fidW, self.input_uf)
        self.pos_num_f = tf.cast(tf.not_equal(self.input_uf, self.user_num), 'float32')
        self.pos_friend = tf.einsum('ab,abc->abc', self.pos_num_f, self.pos_friend)
        self.pos_f = tf.einsum('ac,abc->abc', self.uid_B, self.pos_friend)
        self.pos_f = tf.einsum('ajk,kl->ajl', self.pos_f, self.H_f)
        self.pos_f = tf.reshape(self.pos_f, [-1, max_friend_pu])

    def _pre(self):
        dot = tf.einsum('ac,bc->abc', self.uid_A, self.iidW)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)
        pre = tf.reshape(pre, [-1, self.item_num + 1])
        return pre

    def _create_loss(self):
        self.loss1=self.weight1*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc',self.iidW,self.iidW),0)
                                    *tf.reduce_sum(tf.einsum('ab,ac->abc',self.uid_A,self.uid_A),0)
                                    *tf.matmul(self.H_i,self.H_i,transpose_b=True),0),0)
        self.loss2=self.weight2*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc',self.fidW,self.fidW),0)
                                    *tf.reduce_sum(tf.einsum('ab,ac->abc',self.uid_B,self.uid_B),0)
                                    *tf.matmul(self.H_f,self.H_f,transpose_b=True),0),0)

        self.loss1+=tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.loss2+=tf.reduce_sum((1.0-self.weight2)*tf.square(self.pos_f)-2.0*self.pos_f)

        self.l2_loss0=tf.nn.l2_loss(self.uid_A+self.uid_B)

        self.l2_loss1 = tf.nn.l2_loss(self.WA) + tf.nn.l2_loss(self.BA)+tf.nn.l2_loss(self.HA)
        self.l2_loss2 = tf.nn.l2_loss(self.WB) + tf.nn.l2_loss(self.BB)+tf.nn.l2_loss(self.HB)

        self.loss=self.loss1+self.mu*self.loss2\
                  +self.lambda_bilinear[0]*self.l2_loss0\
                  +self.lambda_bilinear[1]*self.l2_loss1\
                  +self.lambda_bilinear[2]*self.l2_loss2

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.pre=self._pre()



def train_step1(u_batch, y_batch,f_batch):
    """
    A single training step
    """

    feed_dict = {
        deep.input_u: u_batch,
        deep.input_ur: y_batch,
        deep.input_uf:f_batch,
        deep.dropout_keep_prob: 0.3,
    }
    _, loss,wi,wf = sess.run(
        [train_op1, deep.loss,deep.item_w,deep.social_w],
        feed_dict)
    return loss,wi,wf


def dev_step(tset, train_m, test_m, k=100):
    """
    Evaluates model on a dev set

    """
    user_te = np.array(tset.keys())
    user_te2 = user_te[:, np.newaxis]

    ll = int(len(user_te) / 128) + 1

    recall10 = []
    recall50 = []
    recall100 = []
    ndcg10 = []
    ndcg50 = []
    ndcg100 = []

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(user_te))
        u_batch = user_te2[start_index:end_index]

        batch_users = end_index - start_index

        feed_dict = {
            deep.input_u: u_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(
            deep.pre, feed_dict)

        u_b = user_te[start_index:end_index]

        pre = np.array(pre)
        pre=np.delete(pre, -1, axis=1)

        idx = np.zeros_like(pre, dtype=bool)

        idx[train_m[u_b].nonzero()] = True

        pre[idx] = -np.inf

        # recall10

        recall = []

        for kj in [10, 50, 100]:
            idx_topk_part = np.argpartition(-pre, kj, 1)
            pre_bin = np.zeros_like(pre, dtype=bool)
            pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True

            true_bin = np.zeros_like(pre, dtype=bool)
            true_bin[test_m[u_b].nonzero()] = True

            tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
            recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))

        # ndcg10
        ndcg = []

        for kj in [10, 50, 100]:
            idx_topk_part = np.argpartition(-pre, kj, 1)

            topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
            idx_part = np.argsort(-topk_part, axis=1)
            idx_topk = idx_topk_part[np.arange(end_index - start_index)[:, np.newaxis], idx_part]

            tp = 1. / np.log2(np.arange(2, kj + 2))
            test_batch = test_m[u_b]

            DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                              idx_topk].toarray() * tp).sum(axis=1)
            IDCG = np.array([(tp[:min(n, kj)]).sum()
                             for n in test_batch.getnnz(axis=1)])
            ndcg.append(DCG / IDCG)

        recall10.append(recall[0])
        recall50.append(recall[1])
        recall100.append(recall[2])
        ndcg10.append(ndcg[0])
        ndcg50.append(ndcg[1])
        ndcg100.append(ndcg[2])

    recall10 = np.hstack(recall10)
    recall50 = np.hstack(recall50)
    recall100 = np.hstack(recall100)
    ndcg10 = np.hstack(ndcg10)
    ndcg50 = np.hstack(ndcg50)
    ndcg100 = np.hstack(ndcg100)

    print np.mean(recall10), np.mean(ndcg10)
    print np.mean(recall50), np.mean(ndcg50)
    print np.mean(recall100), np.mean(ndcg100)
    f1.write(str(np.mean(recall50))+' '+str(np.mean(ndcg50))+'\n')
    f1.flush()     

    return loss


def get_train_instances1(trset,tfset):
    user_train, item_train,friend_train= [], [],[]
    for i in trset.keys():
        user_train.append(i)
        item_train.append(trset[i])
        friend_train.append(tfset[i])


    user_train = np.array(user_train)
    item_train = np.array(item_train)
    friend_train=np.array(friend_train)
    user_train = user_train[:, np.newaxis]

    return user_train, item_train,friend_train


if __name__ == '__main__':
    np.random.seed(2017)
    random_seed = 2017

    # tp_test =tp4_data

    # tp_train =pd.concat([tp2_data,tp3_data, tp1_data])

    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    u_test = np.array(tp_test['uid'], dtype=np.int32)
    i_test = np.array(tp_test['sid'], dtype=np.int32)
    u_trust = np.array(trust['uid'], dtype=np.int32)
    f_trust = np.array(trust['sid'], dtype=np.int32)

    tset = {}

    count = np.ones(len(u_train))
    train_m = scipy.sparse.csr_matrix((count, (u_train, i_train)), dtype=np.int16, shape=(n_users, n_songs))
    count = np.ones(len(u_test))
    test_m = scipy.sparse.csr_matrix((count, (u_test, i_test)), dtype=np.int16, shape=(n_users, n_songs))

    for i in range(len(u_test)):
        if tset.has_key(u_test[i]):
            tset[u_test[i]].append(i_test[i])
        else:
            tset[u_test[i]] = [i_test[i]]
    trset = {}
    max_item_pu=0
    for i in range(len(u_train)):
        if trset.has_key(u_train[i]):
            trset[u_train[i]].append(i_train[i])
        else:
            trset[u_train[i]] = [i_train[i]]
    for i in trset.keys():
        if len(trset[i])>max_item_pu:
            max_item_pu=len(trset[i])
    print max_item_pu

    for i in trset.keys():
        while len(trset[i]) < max_item_pu:
            trset[i].append(n_songs)

    tfset = {}
    max_friend_pu=0
    for i in range(len(u_trust)):
        if tfset.has_key(u_trust[i]):
            tfset[u_trust[i]].append(f_trust[i])
        else:
            tfset[u_trust[i]] = [f_trust[i]]
    for i in tfset.keys():
        if len(tfset[i])>max_friend_pu:
            max_friend_pu=len(tfset[i])
    print max_friend_pu

    for i in trset.keys():
        if not tfset.has_key(i):
            tfset[i]=[n_users]
        while len(tfset[i]) < max_friend_pu:
            tfset[i].append(n_users)


    batch_size = 128
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = EATNN(n_users, n_songs, 64,32,max_item_pu,max_friend_pu)
            deep._build_graph()
            #optimizer1 = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
             #   deep.loss)
            optimizer1= tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(deep.loss)
            # optimizer1=tf.train.MomentumOptimizer(learning_rate=0.02, momentum=0.95).minimize(deep.loss1)
            train_op1 = optimizer1  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            user_train1, item_train1,friend_train1 = get_train_instances1(trset,tfset)

            for epoch in range(505):
                print epoch
                start_t = _writeline_and_time('\tUpdating...')

                shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
                user_train1 = user_train1[shuffle_indices]
                item_train1 = item_train1[shuffle_indices]
                friend_train1=friend_train1[shuffle_indices]


                ll = int(len(user_train1) / batch_size)
                loss = 0.0

                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train1))

                    u_batch = user_train1[start_index:end_index]
                    i_batch = item_train1[start_index:end_index]
                    f_batch = friend_train1[start_index:end_index]

                    loss1,wi,wf = train_step1(u_batch, i_batch,f_batch)
                    loss += loss1
                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))
                print 'loss1', loss / ll

                if epoch < 500:
                    if epoch % 10 == 0:
                        dev_step(tset, train_m, test_m)
                if epoch >= 500:
                    dev_step(tset, train_m, test_m)























