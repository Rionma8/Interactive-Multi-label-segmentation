import numpy as np
import matplotlib.pyplot as plt

"""
Im : (H x W x 3)
Theta : (Nb_class x H x W)
f : (Nb_class x H x W)
g : (H x W)
Ksi : (Nb_class x H x W x 2)
"""

#### PARAMETERS ####
def Options_IIS(options):
	Opt = [
		('maxIter', 2000),
		('tau_d', 0.5),
		('tau_p', 0.25),
		('alpha', 1.8),
		('sigma', 1.3),
		('gamma', 5),
		('lambda', 120) #0.008)
	]
	for k, v in Opt:
		if k not in options.keys():
			options[k] = v
	return options

###### Pre optimization functions #######

def color_kernel(I,Iij,sigma):
	return np.exp(-.5*np.sum(np.power((I-Iij),2),axis = 2)/sigma**2)/(np.sqrt(2 * np.pi * sigma**2))

def position_kernel(Posx,Posy,Posij,rho_i):
	with np.errstate(divide='ignore',invalid='ignore'):
		pos = np.exp(-.5*(np.power((Posx - Posij[0])/Posx.shape[1],2)+np.power((Posy - Posij[1])/Posx.shape[0],2))/rho_i**2)/(np.sqrt((2*np.pi)*rho_i**2))
	pos[np.isnan(pos)] = 0
	return pos

def compute_dist(Posx,Posy,xj,yj,H,W):
	return np.sqrt(np.power((Posx - xj)/W,2) + np.power((Posy - yj)/H,2))

def compute_f(im,Scribbles,options = {}):
	"""
	image_path : path of the input image of size (H x W)
	Nb_class : number of non-overlapping regions we identify in the image
	return f : (Nb_class x H x W)
	"""
	options = Options_IIS(options)
	H, W = im.shape[0], im.shape[1]
	Nb_class = len(Scribbles)
	Rho = np.zeros((Nb_class,H,W))
	f = np.zeros((Nb_class,H,W))
	P = np.zeros((Nb_class,H,W))
	Posx = np.arange(W)*np.ones((H,W))
	Posy = np.arange(0,H).reshape((H,1)) * np.ones((H,W))
	for i in range(1,Nb_class+1):
		scrib = Scribbles['Class_'+str(i)]
		mi = len(scrib)
		rho_i = np.zeros((mi,H,W))
		for j in range(mi):
			x,y = scrib[j][0], scrib[j][1]
			rho_i[j] = compute_dist(Posx,Posy,x,y,H,W)
		Rho[i-1] = options['alpha']*np.min(rho_i,axis = 0)
		for j in range(mi):
			x,y = scrib[j][0], scrib[j][1]
			color_ker = color_kernel(im,im[y,x],options['sigma'])
			position_ker = position_kernel(Posx,Posy,(x,y),Rho[i-1])
			P[i-1] += color_ker * position_ker
		P[i-1] *= 1/mi

		f[i-1] = (P[i-1]-np.min(P[i-1]))/(np.max(P[i-1])-np.min(P[i-1]))*(1-1e-6) + 1e-6
		f[i-1] = -np.log(f[i-1])
	for i in range(1,Nb_class+1):
		for clas in range(1,Nb_class+1):
			if clas==i: f[i-1][(Rho[clas-1]==0)] = 0
			else:f[i-1][(Rho[clas-1]==0)] = 1000
	return f,Rho

def compute_grad(X):
	"""
	The gradient is computed by forward differences with 
	von Neumann boundary conditions
	X : (H x W)
	grad : (H x W x 2)
	"""
	grad = np.zeros((*X.shape,2))
	grad[:,:-1,0] = X[:,1:] - X[:,:-1]
	grad[:-1,:,1] = X[1:,:] - X[:-1,:]
	return grad

def fct_g(im,options = {}):
	"""
	im : image (H x W x 3)
	g : (H x W) 
	"""
	options = Options_IIS(options)
	gamma = options['gamma']
	Grad_im = np.zeros((im.shape[0],im.shape[1]))
	for i in range(im.shape[2]):
		grad = compute_grad(im[:,:,i])
		Grad_im += np.linalg.norm(grad,ord = 2,axis = 2)**2
	g = np.exp(-gamma*np.sqrt(Grad_im))
	return g

#########################################

##### Tools for optimization ############

def compute_div_Ksi(Ksi):
	"""
	Ksi : (Nb_class x H x W x 2)
	div_Ksi : (Nb_class x H x W)
	"""
	div_ksi = np.zeros(Ksi.shape)
	div_ksi[:,:,0,0] = Ksi[:,:,0,0]
	div_ksi[:,:,1:,0] = Ksi[:,:,1:,0] - Ksi[:,:,:-1,0]

	div_ksi[:,0,:,1] = Ksi[:,0,:,1]
	div_ksi[:,1:,:,1] = Ksi[:,1:,:,1] - Ksi[:,:-1,:,1]

	div_ksi = np.sum(div_ksi,axis = 3)
	return div_ksi

def compute_primal_energy(Theta,f,g,options): 
	lambd = options['lambda']
	grad = np.empty((*Theta.shape,2))
	for i in range(len(Theta)):
		grad[i,:,:,:] = compute_grad(Theta[i,:,:])
	grad = np.linalg.norm(grad,axis = 3)
	#primal_energy = np.sum(Theta*f + lambd*np.reshape(g,(1,*g.shape))*grad)
	primal_energy = np.sum((1/lambd)*Theta*f + lambd*np.reshape(g,(1,*g.shape))*grad)
	return primal_energy

def compute_dual_energy(f,div_Ksi,options):
	lambd = options['lambda']
	#dual_energy = np.sum(np.min(((f + div_Ksi)),axis = 0))
	dual_energy = np.sum(np.min((((1/lambd)*f + div_Ksi)),axis = 0))
	return dual_energy

def Proj_on_Kg(Ksi,g):
	"""
	Projection on Kg = {ksi | |ksi(x)| <= g(x)/2}
	"""
	Projected_Ksi = np.empty(Ksi.shape)
	Norm_ax3_ksi = np.linalg.norm(Ksi,ord = 2, axis = 3)
	Mask1 = (Norm_ax3_ksi < g/2).astype(int)
	Mask1 = np.reshape(Mask1,(*Mask1.shape,1))
	Mask2 = (Norm_ax3_ksi > g/2).astype(int)
	Mask2 = np.reshape(Mask2,(*Mask2.shape,1))
	Projected_Ksi= Ksi*Mask1 + Mask2*(g.reshape((1,*g.shape,1))/2)*(Ksi/Norm_ax3_ksi.reshape((*Norm_ax3_ksi.shape,1)))
	Projected_Ksi[np.isnan(Projected_Ksi)] = 0
	return Projected_Ksi

def Proj_on_B(theta):
	"""
	Projection on B = {Theta | sum(Theta_i) = 1}
	Implementation of the article proposed in the figure 1 of the following article : 
	https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
	"""
	Mu = np.sort(theta, axis = 0)[::-1]
	Mu_cumsum = np.cumsum(Mu,axis = 0)
	J = np.arange(1,Mu.shape[0]+1).reshape((Mu.shape[0],1,1))
	A = Mu - (Mu_cumsum - 1)/J
	B = (A>0).astype(int)
	rho = np.argmin(B,axis = 0)
	rho[np.where(rho==0)] = Mu.shape[0]
	Lag_mult = ((np.sum(B*Mu,axis = 0)-1)/rho).reshape((1,theta.shape[1],theta.shape[2]))
	Projected_theta = (theta - Lag_mult).clip(min=0)
	return Projected_theta


def primal_dual_optimization(f,g,image,options = {}):

	options = Options_IIS(options)
	log_every = options['maxIter']//4
	Nb_class = f.shape[0]


	Theta_init = np.random.rand(*f.shape)

	Ksi = np.zeros((*f.shape,2))

	Primal_energies = []
	Dual_energies  = []
	Gap = []

	Theta_bar = np.copy(Theta_init)
	Theta_t1 = Theta_init

	
	for t in range(options['maxIter']+1):
		
		Theta_t0 = Theta_t1

		#### Update Ksi (dual) #######
		for i in range(Nb_class): 
			Ksi[i] += options['tau_d']*compute_grad(Theta_bar[i])
		Ksi = Proj_on_Kg(Ksi,g)
		###############################

		#### Update Theta (primal) ####
		div_Ksi = compute_div_Ksi(Ksi)
		Theta_t1 = Theta_t0 + options['tau_p']*(div_Ksi - (1/options['lambda'])*f)
		Theta_t1 = Proj_on_B(Theta_t1)
		###############################

		#### Over-relaxation ##########
		Theta_bar = 2*Theta_t1 - Theta_t0
		###############################

		#### Energies #################
		Primal_energies.append(compute_primal_energy(Theta_bar,f,g,options))
		Dual_energies.append(compute_dual_energy(f,div_Ksi,options))
		Gap.append(Primal_energies[-1] - Dual_energies[-1])
		###############################

		if t>100 and (np.abs(Gap[-1]-Gap[-2])) < 0.001*Gap[-2]: break

		if t == 0 or t % log_every == 0:
			print('Iteration = {} ; Dual energy = {} ; Primal Energy = {} ; Gap = {}'.format(t,Dual_energies[-1], Primal_energies[-1], Gap[-1]))
			fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
			axes[0].imshow(image)
			axes[0].axis('off')
			axes[0].set_title('original image')
			axes[1].imshow(result_image_2class(Theta_t1,image))
			axes[1].axis('off')
			axes[1].set_title('segmentation (it. %d)'%( t ))	
			fig.tight_layout()
			plt.pause(0.05)
	

	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
	axes[0].imshow(image)
	axes[0].axis('off')
	axes[0].set_title('original image')
	axes[1].imshow(result_image_2class(Theta_t1,image))
	axes[1].axis('off')
	axes[1].set_title('segmentation random initialization')	
	fig.tight_layout()
	plt.pause(0.05)

	return Theta_t1, Primal_energies, Dual_energies, Gap


######### functions to plot results ###########
def plot_energies(primal_energies,dual_energies,gap):
	plt.subplots(1,2,figsize = (12,8))

	plt.subplot(1,2,1)
	plt.plot(dual_energies,color = 'tab:orange',label = 'Dual energy')
	plt.plot(primal_energies,color = 'tab:blue',label = 'Primal energy')
	plt.legend()

	plt.subplot(1,2,2)
	plt.plot(gap,label = 'Energy Gap')
	plt.legend()
	
	plt.show()

def result_image_2class(Theta,image):
	seg = np.argmax(Theta,axis = 0)
	im_seg = np.copy(image)
	if Theta.shape[0]==2:
		im_seg[seg==0,1:] = 0
		im_seg[seg==1,:-1] = 0
	else:
		for r in range(Theta.shape[0]):
			im_seg[seg==r] = np.random.rand(3)
	return im_seg


def plot_scribbles(im,scribble):
	Nb_class = len(scribble)
	plt.imshow(im)
	plt.axis('off')
	colors = ['tab:blue','tab:orange','black','red','green','yellow']
	for i in range(Nb_class):
		scrib = np.array(scribble['Class_'+str(i+1)]).T
		plt.plot(scrib[0],scrib[1],'o',color = colors[i],markersize =2)
	plt.show()

################################################


