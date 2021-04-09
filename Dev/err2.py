    def get_error_distributions2(self):

        '''
        Gets the error distributions based on a partitioning of the
        Width-ApproximateDistance plane

        u: np.array, shape=(n_samples,)
            Widths of the bounds of the sample pairs, min-max scaled.

        v: np.array, shape=(n_samples,)
            Approximate distance of the sample pairs, min-max scaled.

        self.kmeans: sklearn.cluster.KMeans
            K-Means classifier for partitioning of the uv plane.

        self.cluster_map: np.array, shape=(nx,nx)
            Array containing the partition to which a pair belongs.
            cluster_map[i,j] is the partition for distance ij.

        self.hhs: list of arrays
            List containing the distribution of errors for different partitions.
            hhs[i] is the sorted errors for sample pairs belonging to partition i.

        self.lower: np.array, shape=(nx,nx)
            Array of the approximate distance adjusted by the error distribution
            according to specified min_prob. The idea is that the true distance
            should only have a min_prob probability of being lower than the value
            in this array. We use this to identify candidate pairs to refine.

        self.IJs: list of tuples
            The list of candidate pairs which we will refine, i.e. calculate the
            exact distance.

        '''

        diff = self.G[:,1]-self.G[:,0]  # Width
        LRApprox = self.LR.predict(self.G[:,:2],self.GF) # dhat

        err = self.G[:,2]-LRApprox

        # max-min scaling of the W-dhat plane
        # W -> u
        # dhat -> v
        u=diff.copy()
        v = LRApprox.copy()
        u0 = np.min(u)
        u-=u0
        u1 = np.max(u)
        u/=u1

        v0 = np.min(v)
        v-=v0
        v1 = np.max(v)
        v/=v1

        uv = np.vstack([u,v])
        uv.shape

        def locality(x):
            d = np.linalg.norm(x-uv.T,axis=1)
            l = 0
            r=0.05
            while l<100:
                mask = d<r
                serr = np.sort(err[mask])
                l = len(serr)
                r*=1.1
            return serr[int(l*self.min_prob)]

        X,Y = np.meshgrid(np.linspace(0,1,50),np.linspace(0,1,50))
        XY = np.vstack([X.flatten(),Y.flatten()]).T
        Z = np.array([locality(i) for i in np.vstack([X.flatten(),Y.flatten()]).T])

        self.knnr = KNNR()
        self.knnr.fit(np.vstack([X.flatten(),Y.flatten()]).T,Z)

        self.hs = self.knnr.predict(
                np.vstack(
                [((self.diff.flatten()-u0)/u1),
                 ((self.LRApprox.flatten()-v0)/v1)]
                ).T
              ).reshape(self.nx,self.nx)
        


        #self.min_prob,t,hs = self.adjust_p()
        #print('min_prob reduced to %f' % self.min_prob)

        # Find lower value limit (by min prob)

        # self.lower is the approximate distance less the error according to probability p and error partition
        self.lower = self.LRApprox+self.hs*(self.diff!=0)
        self.lower = np.clip(self.lower,self.Bounds[:,:,0],None)

        # Use lower to identify those points we want to refine (stored in self.IJ)
        II,JJ = np.meshgrid(np.arange(self.nx),np.arange(self.nx))

        mask = (self.lower.T<np.sort(self.LRApprox,axis=1)[:,self.threshold]).T



        igj = II[mask]<JJ[mask]
        a1=np.vstack( [II[mask][igj],JJ[mask][igj]])
        a2=np.vstack( [JJ[mask][~igj],II[mask][~igj]])
        a3 = np.hstack([a1,a2])

        a4 = lexsort_based(a3.T)
        self.IJs = list(map(tuple,a4))
        #print(time.time()-st)


        #IJs = [(i,j) for i,j in zip(II[mask],JJ[mask])]
        #IJs = [(i,j) if i<j else (j,i) for i,j in IJs ]
        #self.IJs = (list(set(IJs)))