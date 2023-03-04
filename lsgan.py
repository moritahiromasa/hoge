import os
import random
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.utils import save_image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image


#param init_setting
workers = 2  #
batch_size = 64 #
nch = 3 # カラーなのでチャンネル数を3に設定 
nz = 100 
nch_g = 32 
nch_d = 32 #
n_epoch = 25 # 学習回数

# 最適化手法の設定
lr = 0.0002  #
beta1 = 0.5
beta2 = 0.5

outd = 'result_lsgan'



try:
	os.makedirs(outd, exist_ok=True)
except OSError as error:
	print(error)
	pass

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



class Generator(nn.Module):
	"""
	生成器のクラス
	"""	
	def __init__(self, nz=nz, nch_g=nch_g, nch=nch):
		# :param nch: カラーなのでチャンネル数は3
		"""
		:param nz:入力ベクトルzの次元
		:param nch_g: 最終層の入力チャンネル数
		:param nch: 出力画像のチャンネル数
		"""
		super(Generator, self).__init__()
		
		#ニューラルネットワークの構造
		self.layers = nn.ModuleDict({
			'layer0':nn.Sequential(
				nn.ConvTranspose2d(nz, nch_g*16, kernel_size=4, stride=1, padding=0),
				nn.BatchNorm2d(nch_g*16),
				nn.ReLU()
			),
			
			'layer1':nn.Sequential(
				nn.ConvTranspose2d(nch_g*16, nch_g*8, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_g*8),
				nn.ReLU()
			),
			
			'layer2':nn.Sequential(
				nn.ConvTranspose2d(nch_g*8, nch_g*4, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_g*4),
				nn.ReLU()
			),
			
			'layer3':nn.Sequential(
				nn.ConvTranspose2d(nch_g*4, nch_g*2, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_g*2),
				nn.ReLU()
			),
			
			'layer4':nn.Sequential(
				nn.ConvTranspose2d(nch_g*2, nch_g, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_g),
				nn.ReLU()
			),
			
			'layer5':nn.Sequential(
				nn.ConvTranspose2d(nch_g, nch, kernel_size=4, stride=2, padding=1),
				nn.Tanh()
			),
		})
	
	def forward(self, z):
		"""
		順方向の演算
		:param z:
		:return :
		"""
		for layer in self.layers.values():

			z = layer(z)
		return z


### Generatorの動作確認 ###

G = Generator(nz=nz, nch_g=nch_g)



class Discriminator(nn.Module):
	"""
	識別器のクラス
	"""
	def __init__(self, nch=nch, nch_d=nch_d):
		"""
		:param nch: 入力画像のチャンネル数
		:param nch_d: 先頭層の出力チャンネル数
		"""
		super(Discriminator, self).__init__()

		#ニューラルネットワークの構造の定義
		self.layers = nn.ModuleDict({
			'layer0':nn.Sequential(
				nn.Conv2d(nch, nch_d, kernel_size=4, stride=2, padding=1), #畳み込み
				nn.LeakyReLU(negative_slope=0.2) #LeakyReLU
			),

			'layer1':nn.Sequential(
				nn.Conv2d(nch_d, nch_d*2, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_d*2),
				nn.LeakyReLU(negative_slope=0.2)
			),

			'layer2':nn.Sequential(
				nn.Conv2d(nch_d*2, nch_d*4, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_d*4),
				nn.LeakyReLU(negative_slope=0.2)
			),

			'layer3':nn.Sequential(
				nn.Conv2d(nch_d*4, nch_d*8, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_d*8),
				nn.LeakyReLU(negative_slope=0.2)
			),

			'layer4':nn.Sequential(
				nn.Conv2d(nch_d*8, nch_d*16, kernel_size=4, stride=2, padding=1),
				nn.BatchNorm2d(nch_d*16),
				nn.LeakyReLU(negative_slope=0.2)
			),

			'layer5': nn.Conv2d(nch_d*16, 1, kernel_size=4, stride=1, padding=0),
		})
	
	def forward(self, x):
		"""
		順方向の演算
		:param x: 本物画像あるいは生成画像
		:return : 識別信号
		"""
		for layer in self.layers.values():
			x = layer(x)

		return x.squeeze()


### Discriminatorの動作確認 ###

D = Discriminator(nch=nch, nch_d=nch_d)

# 入力する乱数
input_z = torch.randn(1, nz)

# ノイズのTensor(B, 100, 1, 1)を画像のTensor(B, 1, 20, 20)に変換
input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)

# 偽画像を出力
fake_images = G(input_z)

print("(G)input_z_shape:", input_z.shape) # 入力の各次元のサイズ
print("(G)fake_images_shape:", fake_images.shape) # 出力の各次元のサイズ

# 偽画像を出力
d_out = D(fake_images)

# 出力d_outにSigmoidを書けて0から1に変換
print(nn.Sigmoid()(d_out))


### DataLoaderの作成 ###
def make_datapath_list():

	# 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する
	
	train_img_list = list() # 画像ファイルパスを格納

	for img_idx in range(1, 100000):
		if img_idx < 10:
			img_path = "/autofs/diamond2/share/users/morita/data/img_align_celeba/00000" + str(img_idx)+'.jpg'
		elif img_idx < 100:
			img_path = "/autofs/diamond2/share/users/morita/data/img_align_celeba/0000" + str(img_idx)+'.jpg'

		elif img_idx < 1000:
			img_path = "/autofs/diamond2/share/users/morita/data/img_align_celeba/000" + str(img_idx)+'.jpg'

		elif img_idx < 10000:
			img_path = "/autofs/diamond2/share/users/morita/data/img_align_celeba/00" + str(img_idx)+'.jpg'
		elif img_idx < 100000:
			img_path = "/autofs/diamond2/share/users/morita/data/img_align_celeba/0" + str(img_idx)+'.jpg'
		train_img_list.append(img_path)
	
	return train_img_list


class ImageTransform():
	""" 画像の前処理 """

	def __init__(self, mean, std):
		self.data_transform = transforms.Compose([
			transforms.CenterCrop(225), # センタークロップいらないと思う
			transforms.RandomCrop(128), # サイズ128でランダムクロップ
#			transforms.RandomHorizonFlip(), # 左右反転、顔認識使うなら必要
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])	
	
	# ここ不明。怪しい。必須らしいが。
	def __call__(self, img):
		return self.data_transform(img)

class GAN_Img_Dataset(Dataset):
	"""画像のDatasetのクラス。PytorchのDatasetクラスを継承"""

	def __init__(self, file_list, transform):
		self.file_list = file_list
		self.transform = transform

	def __len__(self):
		""" 画像の枚数を返す """
		return len(self.file_list)
	
	def __getitem__(self, index):
		""" 前処理した画像のTensor形式のデータを取得 """

		img_path = self.file_list[index]
		img = Image.open(img_path)
		
		# 画像の前処理
		img_transformed = self.transform(img)

		return img_transformed

# DataLoaderの作成と動作確認

# ファイルリストを作成
train_img_list = make_datapath_list()

# Datasetを作成
mean=(0.5, 0.5, 0.5)
std=(0.5, 0.5, 0.5)
train_dataset = GAN_Img_Dataset(file_list=train_img_list,
			transform=ImageTransform(mean, std))
			
train_dataloader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True)

# 動作確認 -> 問題なし
batch_iterator = iter(train_dataloader) # イテレータに変換
images = next(batch_iterator) # 一番目の要素を取り出す
print(images.size()) # torch.size([64, 3, 225, 225]) #センタークロップしているので225になる


"""
重み初期化関数
"""
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		# Conv2dとConvTranspose2dの初期化 
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('BatchNorm') != -1:
		# BatchNorm2dの初期化
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

# 初期化の実施
G.apply(weights_init)
D.apply(weights_init)

print("ネットワーク初期化完了")


# 損失を記録するリストを定義する
Loss_D_list, Loss_G_list = [], []


# モデルを学習させる関数

def train_model(G, D, dataloader, n_epoch):

	# 定期的に画像を保存するためのカウントする変数
	count = 0
	
	# GPUが使えるかを確認
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("使用デバイス:", device)
	
	# 最適化手法の設定
	g_optimizer = torch.optim.Adam(G.parameters(), lr, [beta1, beta2], weight_decay=1e-5)
	d_optimizer = torch.optim.Adam(D.parameters(), lr, [beta1, beta2], weight_decay=1e-5)

	# 誤差関数を定義
	criterion = nn.MSELoss(reduction='mean') # 損失関数は平均二乗誤差

	# パラメータをハードコーティング
	mini_batch_size = 64  # batch_sizeも64
	
	# ネットワークをGPUへ
	G.to(device)
	D.to(device)

	# モデルを訓練モードに
	G.train()
	D.train()

	# ネットワークがある程度固定であれば、高速化させる
	torch.backends.cudnn.benchmark = True

	# 画像の枚数
	num_train_imgs = len(dataloader.dataset)
	batch_size = dataloader.batch_size

	# イテレーションカウントをセット
	iteration = 1
	logs = []
	

	# epochのループ
	for epoch in range(n_epoch):
		
		# 開始時刻を保存
		t_epoch_start = time.time()
		epoch_g_loss = 0.0 # epochの損失和
		epoch_d_loss = 0.0 # epochの損失和


		print('-------------')
		print('Epoch {}/{}'.format(epoch, n_epoch))
		print(' (train)  ')

		# データローダーからminibatchずつ取り出すループ
		for itr, images in enumerate(dataloader):
			
			# 1. Discriminatorの学習
			# ミニバッチがサイズが１だと、バッチノーマライゼーションでエラーになるので避ける
			if images.size()[0] == 1:
				coutinue
				
			# GPUが使えるならGPUにデータを送る
			images = images.to(device)

			# 正解ラベルと偽ラベルを作成
			# epochの最後のインテレーションはミニバッチの数が少なくなる
			mini_batch_size = images.size()[0]
			label_real = torch.full((mini_batch_size, ), 1).to(device)
			label_fake = torch.full((mini_batch_size, ), 0).to(device)

			# 真の画像を判定
			d_out_real = D(images)


			# 偽の画像を生成して判定
			input_z = torch.randn(mini_batch_size, nz).to(device)
			input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
			fake_images = G(input_z)
			
			# 真の画像の判定
			d_out_fake = D(fake_images)
		
			

			### 誤差を計算 ###
			#import ipdb; ipdb.set_trace()
			label_fake = label_fake.type_as(d_out_fake.view(-1)) # 型を揃える
			d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
			
			label_real = label_real.type_as(d_out_real.view(-1)) # 型を揃える
			d_loss_real = criterion(d_out_real.view(-1), label_real)

			d_loss = d_loss_real + d_loss_fake

			# バックプロパケーション
			g_optimizer.zero_grad() 
			d_optimizer.zero_grad()

			d_loss.backward()
			d_optimizer.step()
			

			# 2. Generatorの学習
			# 偽の画像を生成して判定
			input_z = torch.randn(mini_batch_size, nz).to(device)
			input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
			fake_images = G(input_z)
			d_out_fake = D(fake_images)
			

			#### 誤差の計算 ###
			
	#		import ipdb; ipdb.set_trace()
			#g_loss = criterion(d_loss_fake, label_real[0])
			g_loss = criterion(d_out_fake.view(-1), label_real)

			# バックプロバゲーション
			#import ipdb; ipdb.set_trace()
			g_optimizer.zero_grad()
	#		d_optimizer.zero_grad()
			g_loss.backward()
			g_optimizer.step()
			
			# 3. 記録
			epoch_d_loss += d_loss.item()
			epoch_g_loss += g_loss.item()
			iteration += 1
		
			# 定期的に画像を保存

			if itr % 100 == 0:
				save_image(fake_images, './result_lsgan/{0:03d}epoch_{1:04d}.jpg'.format(epoch + 1, count), normalize=True, nrow=8)
				count += 1

		# epochのphaseごとのlossと正解率
		t_epoch_finish = time.time()
		print('-------------')
		print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
			epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
		print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
		
		Loss_D_list.append(epoch_d_loss/batch_size)
		Loss_G_list.append(epoch_g_loss/batch_size)

		t_epoch_start = time.time()
	
	return G, D

# 学習、検証を実行する
G_update, D_update = train_model(
	G, D, dataloader=train_dataloader, n_epoch=n_epoch)


print("Loss_D_list:", Loss_D_list)
print("Loss_G_list:", Loss_G_list)

map1 = ",".join(map(str, Loss_D_list))
map2 = ",".join(map(str, Loss_G_list))

Dloss_list = list(map1)
Gloss_list = list(map2)



# ファイルに損失を書き込み
with open('lsgan_Loss.csv', 'w') as f:
	for i in Dloss_list:
		f.write(i)
	f.write("\n")

	for i in Gloss_list:
		f.write(i)
	f.write("\n")














