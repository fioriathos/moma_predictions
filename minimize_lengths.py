import numpy as np
import recursive_lengths as rl
import imp
import time
imp.reload(rl)
class minimize_lengths(object):
    """Do minimization on """
    def __init__(self, free,fixed={},\
                 method='L-BFGS-B',boundary=None):
        """These 3 dictionary fixed the free param, constrained param, boundary"""
        assert type(fixed)==dict
        assert type(free)==dict;assert free!={}
        self.fixed = fixed; self.free = free
        self.cons=None #constraints
        self.method = method
        self.boundary = boundary
        if boundary==None: # Only positive values allowed
            self.boundary = [(1e-13,None)]*len(free)
        #Check all keys are either fixed or not
        assert set(fixed.keys())|set((free.keys()))==set(('sm2', 'sl2',\
                                                          'gamma','mlam'))
        # Fix them as model parameters
        def set_att(key,val):
            if key=='gamma':self.gamma=val
            if key=='sl2':self.sl2=val
            if key=='sm2':self.sm2=val
            if key=='mlam':self.mlam=val
        #set attributes
        for key,val in free.items():
            set_att(key,val)
        for key,val in fixed.items():
            set_att(key,val)
    def fix_par(self,vec, **kwargs):
        """From np.array vec divide in array the non fixed and dict the fix by giving fixed"""
        from collections import OrderedDict
        vecout = {}
        tmp = OrderedDict([('mlam',vec[0]),('gamma',vec[1]),( 'sl2',vec[2]),( 'sm2',vec[3])])
        for key in kwargs:
            vecout[key]=tmp[key]
            del tmp[key]
        return np.array([tmp[key] for key in tmp]), vecout
    def rebuild_param(self,vec,**kwargs):
        """ Inverse operation than fix_par"""
        from collections import OrderedDict
        tmp = OrderedDict([( 'mlam',None),('gamma',None),( 'sl2',None),( 'sm2',None)])
        for key,val in kwargs.items():
            assert val!=None, "Can't have None as fixed values"
            tmp[key]=val
        for key,val in tmp.items():
            if val==None:
                tmp[key]=vec[0]
                vec = np.delete(vec,0)
        return np.array([tmp[key] for key in tmp])
    def tot_objective(self,x,in_dic,fun=rl.grad_obj_wrap,reg=None):
        """Give obj and grad giving initial conditions and data"""
        obj, grad = fun(x,in_dic)
        if reg is None:
            return obj,grad
        else:
            #Ridge regression for sigmal param i.e. prior normal with variance
            # 1/sqrt(regularize)
            obj += reg*(x[1]**2+x[2]**2)
            grad[1] += 2*reg*x[1]
            grad[2] += 2*reg*x[2]
            return obj, grad 
    def tot_grad_obj(self,x0,in_dic,fun=rl.grad_obj_wrap,reg=None):
        """Return total obj and grad depending on the x0 np.array"""
        # From the reduced x0 rebuild entire vector and compute obj and grad
        #ts = time.time()
        tmp =\
        self.tot_objective(self.rebuild_param(x0,**self.fixed),in_dic,fun,reg)
        obj = tmp[0]
        # return the sliced grad
        grad =self.fix_par(tmp[1], **self.fixed)[0] 
        #print(time.time()-ts)
        #RESCALE LOG LIKELIHOOD IN ORDER TO HAVE MORE SUITABLE NUMBERS
        obj = obj/in_dic['n_point']
        grad = grad/in_dic['n_point']
        return obj,grad.reshape(-1)
    def initialize(self):
        """Return the x np array"""
        x0 = [None]*4
        for i in self.free:
            if i=='mlam':x0[0]=self.free[i]
            if i=='gamma':x0[1]=self.free[i]
            if i=='sl2':x0[2]=self.free[i]
            if i=='sm2':x0[3]=self.free[i]
        x0 = [x for x in x0 if x is not None]
        return np.array(x0)
    def minimize_both_vers(self,in_dic,x0=None,numerical=False,fun=rl.grad_obj_wrap,reg=None,factr=1e4,pgtol=1e-08):
        """Minimize module.tot_grad_obj(t,path) at point x0={mu:,sigmas,..} considering dic['fix]={mu:,..}"""
        from scipy.optimize import fmin_l_bfgs_b
        # Initialize intial condition for first time
        if x0 is None:
            x0 = self.initialize()
        if fun==rl.cost_function:
            assert numerical==True, "no gradient for cost fun"
        if numerical:
            funct = lambda x,y,z:\
                self.tot_grad_obj(x0=x,in_dic=y,fun=z,reg=reg)[0]
            x,obj,tmp = fmin_l_bfgs_b(funct, x0, args=(in_dic,fun,reg),\
                        approx_grad=True,epsilon=1e-08,bounds=self.boundary,factr=factr,pgtol=pgtol)
        else:
            x,obj,tmp = fmin_l_bfgs_b(self.tot_grad_obj,x0,args=(in_dic,fun,reg),\
                        fprime=None,bounds=self.boundary,factr=factr,pgtol=pgtol)
        total_par = self.rebuild_param(x,**self.fixed)
        return tmp,total_par,obj
    def minimize(self,in_dic, x0=None, numerical=False,\
                 fun=rl.grad_obj_wrap,reg=None):
        """Use Analytical gradient until it workds. Then use numerical in case"""
        tmp,total_par,obj  =\
        self.minimize_both_vers(in_dic=in_dic,numerical=numerical,x0=x0,fun=fun,reg=reg)
        #tt = self.tot_objective(total_par,in_dic)
        ret = {}
        ret['log_lik'] = -obj
        ret['message'] = tmp['task']
        ret['status'] = tmp['warnflag']
        ret['jac'] = tmp['grad']
        ret['best_param'] = {'mlam':total_par[0],\
                            'gamma':total_par[1],\
                            'sl2':total_par[2],\
                            'sm2':total_par[3],\
                            }
        if tmp['warnflag']==True or tmp['warnflag']==0:
            self.mlam = total_par[0]
            self.gamma = total_par[1]
            self.sl2= total_par[2]
            self.sm2= total_par[3]
        return ret
    def correct_scaling(self,in_dic):
        res = in_dic['rescale']
        return [self.mlam/res,self.gamma,self.sl2/res**2,self.sm2/res**2]
    def gradient_descent(self,in_dic,eta=1e-06,runtime=10000,x0=None\
                         ,show=False,fun=rl.grad_obj_wrap,reg=None):
        if x0 is None:
            theta = self.initialize()
        else:
            theta=x0
        for k in range(runtime):
            _ , grtheta = self.tot_grad_obj(x0=theta,in_dic=in_dic,\
                              fun=fun,reg=reg)
            vt = eta*grtheta
            theta = theta-vt
            if show:
                print('objective',_,theta)
        return theta,_
    def ADAM(self,in_dic,b1=0.9,b2=0.99,eta=1e-03,eps=1e-10,runtime=10000,x0=None,show=False):
        if x0 is None:
            theta = self.initialize()
        else:
            theta=x0
        mt=st=0
        for k in range(1,runtime):
            _ , gt =  self.tot_grad_obj(x0=theta,in_dic=in_dic,par=True)
            mt =b1*mt+(1-b1)*gt
            st = b2*st+(1-b2)*np.power(gt,2)
            mh = mt/(1-b1**k)
            sh = st/(1-b2**k)
            if show:
                print('objective',_)
            theta = theta-eta*mh/(np.sqrt(sh)+eps)
        return theta,_
 
        #if tmp['success']==False:
        #    print("Probably a problem with gradient, do numerical")
        #    tmp,total_par,lik_grad = self.minimize_both_vers(in_dic=in_dic,x0=tmp['x'],numerical=True)
        #print("--- %s seconds ---" % (time.time() - start_time))
#        self.<mlam>= total_par[1]
#        self.gamma= total_par[2]
#        self.sl2= total_par[3]
#        self.sm2= total_par[4]
#        return tmp,total_par,lik_grad
#    @jit
#    def row_slice(self, xt, nproc):
#        """Return sliced array in nproc times"""
 #       if nproc is None: nproc = self.nproc
#        cs = xt.shape[0]//nproc #chuncksize
#        tmp = [xt[i*cs:cs*i+cs,:] for i in range(nproc)]
#        if nproc*cs != xt.shape[0]:
#            tmp[-1] = np.concatenate((tmp[-1],xt[nproc*cs:xt.shape[0],:]),axis=0)
#        return tmp
#    def minimize_with_restart(self,num_restart,nproc=None):
#        """Do the minimization with num_restart times using normal dist with
#        0.01 CV"""
#        def minimize_wrapper(matx0):
#            return [self.minimize(matx0[i,:]) for i in range(matx0.shape[0])]
#
#        x0 = self.initialize()[:,None]
#        x0s = np.apply_along_axis(lambda x:\
#                               0.01*x*randn()+x,axis=0,\
#                                arr=np.repeat(x0,num_restart,axis=1)).T
#        slx0s = self.row_slice(x0s,nproc)
#        pool = Pool(processes=nproc)
#        outputs = [pool.amap(minimize_wrapper,[x0]) for x0 in slx0s]
#        outputs = [o.get() for o in outputs]
#        return outputs
#    def heat_map_variables(self,cent_val,folds,step):
#        # Create sampling for heat map
#        import math
#        log2 = lambda x: math.log(x, 2.0)
#        ran = lambda x: x[1]*2**(np.arange(-log2(x[0]),log2(x[0])+step,step)) #how to create the sampling for the heat map
#        #The natural system for the values
#            cent_val[1] = 1./cent_val[1]
#            cent_val[2:] = np.sqrt(cent_val[2:])
#        valus =[self.mu,1./self.gamma ,np.sqrt(self.sigmas)\
#                ,np.sqrt(self.gstds) ]
#        self.mus,self.gammas,self.sigmass,self.gstdss = map(ran,zip(folds,valus))
#        return [(mu,gamma,sigmas,gstds) for mu in self.mus for gamma in\
#                1./self.gammas for sigmas in self.sigmass**2 for gstds in\
#                self.gstdss**2]
#    def heat_map_variables(self,heatmapvar):
#        assert  type(heatmapvar['mu']) == np.ndarray and \
#                type(heatmapvar['sigmas']) == np.ndarray and \
#                type(heatmapvar['gstds']) == np.ndarray and \
#                type(heatmapvar['gamma']) == np.ndarray
#        print( "EASIER TO THINK ABOUT TAU (1/gamma) , SIGMA ADN GSTD (no\
#            square) RANGE FOR A GOOD HEATMAP!!!")
#        return [(mu,gamma,sigmas,gstds) for mu in heatmapvar['mu'] for gamma in\
#                heatmapvar['gamma'] for sigmas in heatmapvar['sigmas']  for gstds in\
#                heatmapvar['gstds']]
#    def heat_map(self,heatmapvar):
#        """Return the heat map"""
#        pool = Pool(processes=self.nproc)
#        var = self.heat_map_variables(heatmapvar)
#        def func(x):
#            return self.tot_objective(x)[0],x
#        output =[ pool.amap(func,[x]) for x in var]
#        outputs = [o.get() for o in output]
#        outputs = [ [x[0],x[1][0],x[1][1],x[1][2],x[1][3]] for j in outputs for x in j ]
#        return np.array(outputs)
#    def errorbars(self, normalized=False):
#        """Return the inverse of the covariance function"""
#        print( "make sure parameters at minimum")
#        print( "mu",self.mu,"gamma",self.gamma,\
#                'sigmas',self.sigmas,'gstds',self.gstds)
#        gpy = self.return_model()
#        H = gpy.multiproc_hessian(self.time,self.path,self.nproc,\
#                                  self.lamb1,self.lamb2,normalized)
#        cov = gpy.covariance(H)
#        err = gpy.errorbars(cov)
#        return {'relmu':err['errmu']/self.mu,\
#                'relgam':err['errgam']/self.gamma,\
#                'relsigmas':err['errsigs']/self.sigmas,\
#                'relgstds':err['errgstds']/self.gstds}
#    def hessian(self,normalized=False):
#        gpy = self.return_model()
#        H = gpy.multiproc_hessian(self.time,self.path,self.nproc,\
#                                  self.lamb1,self.lamb2,normalized)
#        return H 
#    def predict(self,alpha):
#        """predict paths with used alpha scaling"""
#        gpy = self.return_model()
#        #time path error
#        return gpy.parallel_predict(self.time,self.path,self.lamb1,self.lamb2,nproc=8,dt=0.5)
