# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# #############  Multi-task rnn model ####################### #

##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
class AppPreLocPreUserIdenGtr(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenGtr, self).__init__()
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.app_size = parameters.app_size
		self.app_emb_size = parameters.app_encoder_size
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda
		self.rnn_type = parameters.rnn_type
		self.app_emd_mode = parameters.app_emb_mode

		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_app = nn.Embedding(self.app_size, self.app_emb_size)
		self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)

		input_size = self.tim_emb_size*2 + self.app_emb_size + self.loc_emb_size
		if self.rnn_type == 'GRU':
			self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'LSTM':
			self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
		elif self.rnn_type == 'RNN':
			self.rnn = nn.RNN(input_size, self.hidden_size, 1)
		self.init_weights()
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc_uid = nn.Linear(self.hidden_size, self.uid_size)
		self.dec_app = nn.Linear(self.hidden_size, self.app_size)
		self.fc_loc = nn.Linear(self.hidden_size, self.loc_size)

	def init_weights(self):
		ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
		hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
		b = (param.data for name, param in self.named_parameters() if 'bias' in name)

		for t in ih:
			nn.init.xavier_uniform(t)
		for t in hh:
			nn.init.orthogonal(t)
		for t in b:
			nn.init.constant(t, 0)
			
	def forward(self, tim, app, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		c1 = Variable(torch.zeros(1, 1, self.hidden_size))
		app_emb = Variable(torch.zeros(len(app), self.app_emb_size))
		if self.use_cuda:
			h1 = h1.cuda()
			c1 = c1.cuda()
			app_emb = app_emb.cuda()

		tim_emb = self.emb_tim(tim)
		ptim_emb = self.emb_tim(ptim)
		loc_emb = self.emb_loc(loc)

		if self.app_emd_mode == 'sum':
			app_emb = app.mm(self.emb_app.weight)
		# elif self.app_emd_mode == 'avg':
		# 	app_emb = torch.div(app.mm(self.emb_app.weight),torch.sum(app,1).view(len(app),1))
		app_emb = app_emb.unsqueeze(1)

		x = torch.cat((tim_emb, app_emb, ptim_emb, loc_emb), 2)
		x = self.dropout(x)

		if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
			out, h1 = self.rnn(x, h1)
		elif self.rnn_type == 'LSTM':
			out, (h1, c1) = self.rnn(x, (h1, c1))
		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = self.dec_app(out)
		app_score = F.sigmoid(app_out)
		
		loc_out = self.fc_loc(out)
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss
		
		user_out = self.fc_uid(out)
		user_score = F.log_softmax(user_out)

		return app_score,loc_score,user_score

class AppPreLocPreUserIdenGtrLinear(nn.Module):
	"""baseline rnn model, location prediction """
	def __init__(self, parameters):
		super(AppPreLocPreUserIdenGtrLinear, self).__init__()
		self.tim_size = parameters.tim_size
		self.app_size = parameters.app_size
		self.loc_size = parameters.loc_size
		self.uid_size = parameters.uid_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda

		input_size = self.tim_size*2 + self.app_size + self.loc_size
		output_size = self.app_size + self.loc_size + self.uid_size
		self.dropout = nn.Dropout(p=parameters.dropout_p)
		self.fc = nn.Linear(input_size, output_size)

			
	def forward(self, tim, app, loc, uid, ptim):
		x = torch.cat((tim, app, loc, ptim), 1)

		out = self.fc(x)
		out = self.dropout(out)

		out = out.squeeze(1)
		out = F.selu(out)
		
		app_out = out[:,:self.app_size]
		app_score = F.sigmoid(app_out)
		
		loc_out = out[:,self.app_size:self.app_size+self.loc_size]
		loc_score = F.log_softmax(loc_out)  # calculate loss by NLLoss
		
		user_out = out[:,self.app_size+self.loc_size:]
		user_score = F.log_softmax(user_out)

		return app_score,loc_score,user_score

# ############# Define Loss ####################### #
# USE THIS LOSS
class AppLocUserLoss(nn.Module):
	def __init__(self,parameters):
		super(AppLocUserLoss, self).__init__()
		self.alpha = parameters.loss_alpha
		self.beta = parameters.loss_beta
	def forward(self, app_scores, app_target, loc_scores, loc_target, uid_scores, uid_target):
		app_loss = nn.BCELoss()
		loc_loss = nn.NLLLoss()
		uid_loss = nn.NLLLoss()
		loss_app = app_loss(app_scores, app_target)
		loss_loc = loc_loss(loc_scores, loc_target)
		loss_uid = uid_loss(uid_scores, uid_target)
		return loss_app + self.alpha*loss_loc + self.beta*loss_uid, loss_app, loss_loc, loss_uid

# ############# Context embedding ####################### #
class Line_1st(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_1st, self).__init__()
		self.order = 1
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.emb(x2)
		x = w * torch.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u]
		v2 = self.emb.weight[v]
		return v1.dot(v2)/(norm(v1)*norm(v2))

class Line_2nd(nn.Module):
	def __init__(self, num_nodes, emb_size=64):
		super(Line_2nd, self).__init__()
		self.order = 2
		self.emb_size = emb_size
		self.num_nodes = num_nodes
		self.emb = nn.Embedding(num_nodes, emb_size)
		self.ctx = nn.Embedding(num_nodes, emb_size) # context vector

	def forward(self, x1, x2, w):
		x1 = self.emb(x1)
		x2 = self.ctx(x2)
		x = w * torch.sum(x1*x2, dim=1)
		return -F.logsigmoid(x).mean()

	def similarity(self, u, v):
		v1 = self.emb.weight[u].data.cpu().numpy()
		v2 = self.emb.weight[v].data.cpu().numpy()
		return v1.dot(v2)/(norm(v1)*norm(v2))

class Line:
	def __init__(self, line_1st, line_2nd, alpha=2, name='0'):
		self.alpha = alpha
		emb1 = line_1st.emb.weight
		emb2 = line_2nd.emb.weight * self.alpha
		self.embedding = torch.cat((emb1, emb2),1)
		self.name = name

	def similarity(self, u, v):
		v1 = self.embedding[u].data.cpu().numpy()
		v2 = self.embedding[v].data.cpu().numpy()
		return v1.dot(v2)/(norm(v1)*norm(v2))

	def save_emb(self):
		np.save('Line_'+self.name+'.npy',self.embedding.data.cpu().numpy())
		print('********save************')

# ############# Joint training ####################### #
class LocPreUserGtrLocEmb(nn.Module):
	"""baseline rnn model, location prediction with LINE as embedding """

	def __init__(self, parameters, line_1st, line_2nd, alpha=2):
		super(LocPreUserGtrLocEmb, self).__init__()
		self.loc_size = parameters.loc_size
		self.loc_emb_size = parameters.loc_emb_size*2
		self.tim_size = parameters.tim_size
		self.tim_emb_size = parameters.tim_emb_size
		self.uid_size = parameters.uid_size
		self.uid_emb_size = parameters.uid_emb_size
		self.hidden_size = parameters.hidden_size
		self.dropout_p = parameters.dropout_p
		self.use_cuda = parameters.use_cuda

		self.alpha = alpha
		self.emb_loc1 = line_1st.emb
		self.emb_loc2 = line_2nd.emb
		#self.emb_loc = torch.cat((self.emb_loc1, self.emb_loc2 * self.alpha),1)
		self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
		self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

		input_size = self.loc_emb_size + self.tim_emb_size*2
		self.rnn = nn.GRU(input_size, self.hidden_size, 1)
		self.fc = nn.Linear(self.hidden_size + self.uid_emb_size, self.loc_size)

	def forward(self, tim, loc, uid, ptim):
		h1 = Variable(torch.zeros(1, 1, self.hidden_size))
		if self.use_cuda:
			h1 = h1.cuda()

		loc_emb1 = self.emb_loc1(loc)
		loc_emb2 = self.emb_loc2(loc)
		loc_emb = torch.cat((loc_emb1, loc_emb2 * self.alpha),2)
		ptim_emb = self.emb_tim(ptim)
		tim_emb = self.emb_tim(tim)
		x = torch.cat((tim_emb, loc_emb), 2)
		x = torch.cat((x, ptim_emb), 2)
		x = F.dropout(x, p=self.dropout_p)

		out, h1 = self.rnn(x, h1)
		out = out.squeeze(1)
		out = F.selu(out)

		uid_emb = self.emb_uid(uid).repeat(loc.size()[0], 1)
		out = torch.cat((out, uid_emb), 1)

		y = self.fc(out)
		score = F.log_softmax(y)  # calculate loss by NLLoss
		return score
		
	def save_emb(self):
		emb_loc = torch.cat((self.emb_loc1.weight, self.emb_loc2.weight * self.alpha),1)
		np.save('Line_LocPreUserGtrLocEmb.npy',emb_loc.data.cpu().numpy())
		print('********save************')

