import sys
import os
import time

BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
import numpy as np
import random
torch.set_num_threads(2)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

		input_data = data_generator.input_data(args=self.args)
		self.input_data = input_data

		if self.args.train_test_label == 2:
			input_data.het_walk_restart()
			print("neighbor set generation finish")

		feature_list = [input_data.p_title_embed,\
		input_data.p_v_net_embed, input_data.p_a_net_embed, input_data.p_ref_net_embed,\
		input_data.p_net_embed, input_data.a_net_embed, input_data.a_text_embed,\
		input_data.v_net_embed, input_data.v_text_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()

		a_neigh_list_train = input_data.a_neigh_list_train
		p_neigh_list_train = input_data.p_neigh_list_train
		v_neigh_list_train = input_data.v_neigh_list_train

		a_train_id_list = input_data.a_train_id_list
		p_train_id_list = input_data.p_train_id_list
		v_train_id_list = input_data.v_train_id_list

		self.model = tools.HetAgg(args, feature_list, a_neigh_list_train, p_neigh_list_train, v_neigh_list_train, \
								  a_train_id_list, p_train_id_list, v_train_id_list)

		if self.gpu:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay = args.weight_decay)
		self.model.init_weights()

	def model_train(self):
		print('Model Training')
		self.model.train()
		mini_batch_s = self.args.mini_batch_s
		embed_d = self.args.embed_d

		for iter_i in range(self.args.train_iter_n):
			# print('iteration ' + str(iter_i) + ' ...')
			start_time = time.time()
			triple_list = self.input_data.sample_het_walk_triple()

			min_len = 1e10
			for ii in range(len(triple_list)):
				if len(triple_list[ii]) < min_len:
					min_len = len(triple_list[ii])
			batch_n = int(min_len / mini_batch_s)

			loss_list = []
			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])

				for triple_index in range(len(triple_list)):
					triple_list_temp = triple_list[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s] #batch

					c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

					c_out[triple_index] = c_out_temp
					p_out[triple_index] = p_out_temp
					n_out[triple_index] = n_out_temp

				loss = tools.cross_entropy_loss(c_out, p_out, n_out, embed_d)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

				# if (k+1) % 600 == 0:
				# 	print("loss: " + str(loss))
				loss_list.append(loss)

			train_loss = sum(loss_list) / len(loss_list)

			if iter_i+1 % args.save_model_freq == 0:
				model_save_path = 'model_save/'
				if not os.path.isdir(model_save_path):
					os.mkdir(model_save_path)
				save_path = model_save_path + "SIHDGNN_" + str(iter_i) + ".pt"
				torch.save(self.model.state_dict(), save_path)
				print('Model saved at {}'.format(save_path))
				triple_index = 9
				_1,_2,_3 = self.model([], triple_index)

			print('Iteration: {} \t Train_Loss: {:.3f} \t Time: {:.3f}s' \
				  .format(iter_i, train_loss, time.time() - start_time))


if __name__ == '__main__':
	args = read_args()
	# print("------arguments-------")
	# for k, v in vars(args).items():
	# 	print(k + ': ' + str(v))

	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	model_object = model_class(args)

	if args.train_test_label == 0:
		model_object.model_train()

