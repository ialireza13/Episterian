// g++ vfinonedis.cc -o vfino;./vfino

#include<iostream>
#include<fstream>
#include<cmath>
#include<stdlib.h>
#include<iomanip>

using namespace std;

#define N 900//Number of particles
// tehran pop: 8e6
#define L sqrt(1400)//Particles move in a 2D space of size LxL
// tehran L: sqrt(730e3)
#define s 1 //Number of different realizations
#define pi 3.141592653589793
#define d 1.5
// covid19 d: 1.5
#define d2 d*d


double d_ran(){//Returns a random number in [0,1)
	double u=rand()/(RAND_MAX+1.0);
	return u;
}

int i_ran(int max){//Returns integer random number in [0,max]
	return rand()%(max+1);
}




int main(){
	double r[N][2];/*Position vector for each particle*/
	bool A[N],a[N],sA[N];//State of each particle, sA:susceptible, A: infected, a: recovered
	bool upA[N];
	double dist,p,u,x;/*p=Probability of being infected for particles in state S*/
	int t=0;/*Time*/
	int i,j,k,l,m;
	ofstream dif("results.gnumeric");
	int Sinf,count;
	int infA[N],numinfA,susA[N],numsusA,cA[N],numU,U[N];//Useful lists of particles
	double v[N],psi,vv[N][2];//Velocity
	double avdeg=(N-1)*pi*d2/(L*L);//Average degree for a 2D random geometric graph with interaction range d
	double probs[N];//probs[i] is the probability that a susceptible gets infected under i exposures
	double dx,dy;
	double rho,rhoa;
	int ca;
	bool plot=0;
	
	// Assign velocities
	int subGroups = 3;
	int lastN = 0;
	for(int group=1; group<subGroups; group++){
		for(i=lastN; i<int(group*N/subGroups); i++){
			v[i] = group*3.0/subGroups * d;
		}
		lastN = int(group*N/subGroups);
//		cout<<v[i-1]<<'\t';
	}
	for(i=lastN; i<N; i++){
		v[i] = 3.0 * d;
	}
//	cout<<v[i-1]<<'\n';
	
	double gamma = 1.0/18.0; // rec
	double R0 = 2.6;
	double sigma = 1.0/5.2; // inf
	double beta = R0*gamma; // exp
	
	double time_norm = 24*60.0;
	sigma /= time_norm;
	gamma /= time_norm;
	beta /= time_norm;
	
	int I0=1;
	ca=0;//Number of observed macroscopic outbreaks
	rhoa=0.;//Size of observed macroscopic outbreaks
	
	probs[0]=0;
	for(i=1;i<N;i++)probs[i]=sigma;
//	for(i=0;i<N;i++)probs[i]=1.-probs[i];

	for(l=0;l<s;l++){//Different realizations

		//Initial condition: one infected particle
		for(i=0;i<N;i++){
			A[i]=0;
			a[i]=0;
			sA[i]=1;
			upA[i]=0;
			cA[i]=0;
		}
		for(j=0; j<I0; j++){
			i=i_ran(N-1);
			if(A[i]==0){
				A[i]=1;
				sA[i]=0;
			}else j--;
		}
		/*Make the list of particles in each state (no need to track recovered)*/
		numinfA=0;
		numsusA=0;
		for(i=0;i<N;i++){
			infA[numinfA]=i;
			numinfA=numinfA+A[i];
			susA[numsusA]=i;
			numsusA=numsusA+sA[i];
			/*Initial condition for position vector*/
			r[i][0]=d_ran()*L;
			r[i][1]=d_ran()*L;
			psi=2*pi*d_ran();
			vv[i][0]=v[i]*cos(psi);
			vv[i][1]=v[i]*sin(psi);
		}
		t=0;
		while(numinfA!=0){/*We let the system evolve until absorbing configuration is reached*/
			dif<<t<<'\t'<<1.0*numsusA/N<<'\t'<<t<<'\t'<<1.0*numinfA/N<<'\t'<<t<<'\t'<<1.0-(1.0*numinfA/N+1.0*numsusA/N)<<endl;
//			cout<<t<<'\t'<<1.0*numsusA/N<<'\t'<<1.0*numinfA/N<<endl;
			t++;
			//1)Contacts infA-susA
			for(i=0;i<numinfA;i++){
				k=infA[i];
				for(j=0;j<numsusA;j++){
					m=susA[j];
					dx=abs(r[k][0]-r[m][0]);
					dx=fmin(dx,L-dx);
					dy=abs(r[k][1]-r[m][1]);
					dy=fmin(dy,L-dy);
					dist=dx*dx+dy*dy;
					u=d_ran();
					cA[m]=cA[m]+(d2<=dist)*(u<=beta);
				}
			}
			/*Update the states*/
			numU=0;
			for(i=0;i<numsusA;i++){//Make a list of the susceptible particles which are exposed to the infection
				U[numU]=susA[i];
				numU=numU+(bool)cA[susA[i]];
			}
			for(i=0;i<numU;i++){//Only the susceptible which are exposed can get infected
				j=U[i];
				u=d_ran();
				upA[j]=(bool)(u<=probs[cA[j]]);
				A[j]=upA[j];
				sA[j]=!upA[j];
				cA[j]=0;
				upA[j]=0;
			}
			for(i=0;i<numinfA;i++){//The infected particles recover
				j=infA[i];
				u=d_ran();
				a[j]=(u<gamma);
				A[j]=1-a[j];
			}
			
			/*State updated*/
			
			/*Make the list of particles in each state (we do not track recovered particles) and update position*/
			numinfA=0;
			numsusA=0;
			for(i=0;i<N;i++){
				infA[numinfA]=i;
				numinfA=numinfA+A[i];
				susA[numsusA]=i;
				numsusA=numsusA+sA[i];
				/*Position update, with periodic boundary conditions*/
				r[i][0]=r[i][0]+vv[i][0]+L;/*As |v|<L, adding L we make r[i][0]>0*/
				r[i][1]=r[i][1]+vv[i][1]+L;
				r[i][0]=r[i][0]-L*((int)(r[i][0]/L));/*If r[i][0]>L, we assign it r[i][0] mod L*/
				r[i][1]=r[i][1]-L*((int)(r[i][1]/L));
			}
		} // while

		//Absorbing configuration has been reached, we can measure the order parameter
		Sinf=0;
		for(i=0;i<N;i++){
			Sinf=Sinf+(a[i]);
		}
		rho=(double)Sinf/N;
		if(rho>0.02){//There is a macroscopic outbreak
			ca++;
			rhoa=rhoa+rho;
		}
		cout<<l<<endl;
	} // realizations
	plot=plot||(bool)((int)(((double)ca/s)/0.02));//We only plot the average macroscopic outbreak size when the fraction of macroscopic outbreaks is significant
	if(plot==1)
//		dif<<std::setprecision(7)<<p*avdeg<<"\t"<<v/d<<"\t"<<rhoa/ca<<"\t"<<(double)ca/s<<endl;
		cout<<std::setprecision(7)<<p*avdeg<<"\t"<<rhoa/ca<<"\t"<<(double)ca/s<<endl;
	else
//		dif<<std::setprecision(7)<<p*avdeg<<"\t"<<v/d<<"\t"<<"0"<<"\t"<<(double)ca/s<<endl;
		cout<<std::setprecision(7)<<p*avdeg<<"\t"<<"0"<<"\t"<<(double)ca/s<<endl;
	dif.close();
  	return 0;
  }
