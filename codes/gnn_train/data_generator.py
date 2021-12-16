import sys
import os
BASE_PATH = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(BASE_PATH)
import pickle
import re
import random
import config


class input_data(object):
	def __init__(self, args):
		self.args = args

		with open(config.gnn_pre_file, 'rb') as f:
			a_p_list_train, p_a_list_train, p_p_cite_list_train, v_p_list_train, \
			a_neigh_list_train_top, p_neigh_list_train_top, v_neigh_list_train_top, \
			p_title_embed, p_v_net_embed, p_a_net_embed, p_ref_net_embed, p_net_embed, \
			a_net_embed, a_text_embed, v_net_embed, v_text_embed, \
			a_train_id_list, p_train_id_list, v_train_id_list = pickle.load(f)

		print('Load pre_file.pkl')
		self.a_p_list_train = a_p_list_train
		self.p_a_list_train = p_a_list_train
		self.p_p_cite_list_train = p_p_cite_list_train
		self.v_p_list_train = v_p_list_train

		self.p_title_embed = p_title_embed
		self.p_v_net_embed = p_v_net_embed
		self.p_a_net_embed = p_a_net_embed
		self.p_ref_net_embed = p_ref_net_embed
		self.p_net_embed = p_net_embed
		self.a_net_embed = a_net_embed
		self.a_text_embed = a_text_embed
		self.v_net_embed = v_net_embed
		self.v_text_embed = v_text_embed

		self.a_neigh_list_train = a_neigh_list_train_top
		self.p_neigh_list_train = p_neigh_list_train_top
		self.v_neigh_list_train = v_neigh_list_train_top

		self.a_train_id_list = a_train_id_list
		self.p_train_id_list = p_train_id_list
		self.v_train_id_list = v_train_id_list

		self.triple_sample_p = [0.003, 0.004, 0.007, 0.004, 0.004, 0.009, 0.0075, 0.009, 0.017]
		# print("triple_sample_p", triple_sample_p[0], triple_sample_p[1], triple_sample_p[2],triple_sample_p[3], triple_sample_p[4], triple_sample_p[5],triple_sample_p[6], triple_sample_p[7], triple_sample_p[8])


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = config.A_n
		P_n = config.P_n
		V_n = config.V_n

		total_triple_n = [0.0] * 9
		het_walk_f = open(config.het_random_walk, "r")

		for line in het_walk_f:
			line = line.strip()
			path = re.split(' ', line)
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[0] += 1
								elif neighNode[0] == 'p':
									total_triple_n[1] += 1
								elif neighNode[0] == 'v':
									total_triple_n[2] += 1
					elif centerNode[0] == 'p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[3] += 1
								elif neighNode[0] == 'p':
									total_triple_n[4] += 1
								elif neighNode[0] == 'v':
									total_triple_n[5] += 1
					elif centerNode[0] == 'v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a':
									total_triple_n[6] += 1
								elif neighNode[0] == 'p':
									total_triple_n[7] += 1
								elif neighNode[0] == 'v':
									total_triple_n[8] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / total_triple_n[i]
		print("sampling ratio computing finish.")

		return total_triple_n

	def sample_het_walk_triple(self):
		triple_list = [[] for k in range(9)]
		window = self.args.window
		walk_L = self.args.walk_L
		A_n = self.args.A_n
		P_n = self.args.P_n
		V_n = self.args.V_n
		triple_sample_p = self.triple_sample_p
		het_walk_f = open(config.het_random_walk, "r")

		for line in het_walk_f:
			line = line.strip()
			path = re.split(' ', line)
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'a':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[0]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									# random negative sampling get similar performance as noise distribution sampling
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[1]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[1].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[2]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[2].append(triple)
					elif centerNode[0]=='p':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[3]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[3].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[4]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[4].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[5]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[5].append(triple)
					elif centerNode[0]=='v':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'a' and random.random() < triple_sample_p[6]:
									negNode = random.randint(0, A_n - 1)
									while len(self.a_p_list_train[negNode]) == 0:
										negNode = random.randint(0, A_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[6].append(triple)
								elif neighNode[0] == 'p' and random.random() < triple_sample_p[7]:
									negNode = random.randint(0, P_n - 1)
									while len(self.p_a_list_train[negNode]) == 0:
										negNode = random.randint(0, P_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[7].append(triple)
								elif neighNode[0] == 'v' and random.random() < triple_sample_p[8]:
									negNode = random.randint(0, V_n - 1)
									while len(self.v_p_list_train[negNode]) == 0:
										negNode = random.randint(0, V_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[8].append(triple)
		het_walk_f.close()
		return triple_list