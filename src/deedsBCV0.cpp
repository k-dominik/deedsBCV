#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
#include <string.h>
#include <sstream>
#include <x86intrin.h>
#include <pthread.h>
#include <thread>
#include <cstddef>   
#include "zlib.h"
#include <sys/stat.h>

using namespace std;

//some global variables
int RAND_SAMPLES; //will all be set later (if needed)
int image_m; int image_n; int image_o; int image_d=1;
float SSD0=0.0; float SSD1=0.0; float SSD2=0.0; float distfx_global; float beta=1;
//float SIGMA=8.0;
int qc=1;

//struct for multi-threading of mind-calculation
struct mind_data{
	float* im1;
    float* d1;
    uint64_t* mindq;
    int qs;
    int ind_d1;
};

struct parameters{
    float alpha; int levels; bool segment,affine,rigid;
    vector<int> grid_spacing; vector<int> search_radius;
    vector<int> quantisation;
    string fixed_file,moving_file,output_stem,moving_seg_file,affine_file,deformed_file;
};

#include "imageIOgzType.h"
#include "transformations.h"
#include "primsMST.h"
#include "regularisation.h"
#include "MINDSSCbox.h"
#include "dataCostD.h"
#include "parseArguments.h"
#include "deeds_config.h"


int main (int argc, char * const argv[]) {
    
    //PARSE INPUT ARGUMENTS
    
    if(argc<4||argv[1][1]=='h'){
        cout<<"=============================================================\n";
        cout<<"deedsBCV0 v" << DEEDS_VERSION_MAJOR << "." << DEEDS_VERSION_MINOR << "." << DEEDS_VERSION_PATCH << "\n";
        cout<<"Usage (required input arguments):\n";
        cout<<"./deedsBCV -F fixed.nii.gz -M moving.nii.gz -O output\n";
        cout<<"optional parameters:\n";
        cout<<" -a <regularisation parameter alpha> (default 1.6)\n";
        cout<<" -l <number of levels> (default 5)\n";
        cout<<" -G <grid spacing for each level> (default 8x7x6x5x4)\n";
        cout<<" -L <maximum search radius - each level> (default 8x7x6x5x4)\n";
        cout<<" -Q <quantisation of search step size> (default 5x4x3x2x1)\n";
        cout<<" -S <moving_segmentation.nii> (short int)\n";
        cout<<" -A <affine_matrix.txt> \n";
        cout<<"=============================================================\n";
        return 1;
    }
    parameters args;
    //defaults
    args.grid_spacing={8,7,6,5,4};
    args.search_radius={8,7,6,5,4};
    args.quantisation={5,4,3,2,1};
    args.levels=5;
    parseCommandLine(args, argc, argv);

    size_t split_fixed=args.fixed_file.find_last_of("/\\");
    if(split_fixed==string::npos){
        split_fixed=-1;
    }
    size_t split_moving=args.moving_file.find_last_of("/\\");
    if(split_moving==string::npos){
        split_moving=-1;
    }
    
    
    if(args.fixed_file.substr(args.fixed_file.length()-2)!="gz"){
        cout<<"images must have nii.gz format\n";
        return -1;
    }
    if(args.moving_file.substr(args.moving_file.length()-2)!="gz"){
        cout<<"images must have nii.gz format\n";
        return -1;
    }

    cout<<"Starting registration of "<<args.fixed_file.substr(split_fixed+1)<<" and "<<args.moving_file.substr(split_moving+1)<<"\n";
    cout<<"=============================================================\n";

    string outputflow;
    outputflow.append(args.output_stem);
    outputflow.append("_displacements.dat");
    string outputfile;
    outputfile.append(args.output_stem);
    outputfile.append("_deformed.nii.gz");
    
                      
    //READ IMAGES and INITIALISE ARRAYS
    
    chrono::time_point<chrono::high_resolution_clock> time1,time2,time1a,time2a;

	RAND_SAMPLES=1; //fixed/efficient random sampling strategy
	
	float* im1; float* im1b;
    
	int M,N,O,P; //image dimensions
    
    //==ALWAYS ALLOCATE MEMORY FOR HEADER ===/
	char* header=new char[352];
    
	readNifti(args.fixed_file,im1b,header,M,N,O,P);
    image_m=M; image_n=N; image_o=O;

	readNifti(args.moving_file,im1,header,M,N,O,P);
	

    if(M!=image_m|N!=image_n|O!=image_o){
        cout<<"Inconsistent image sizes (must have same dimensions)\n";
        return -1;
    }
	
	int m=image_m; int n=image_n; int o=image_o; int sz=m*n*o;
    
    //assume we are working with CT scans (add 1024 HU)
    float thresholdF=-1024; float thresholdM=-1024;

    for(int i=0;i<sz;i++){
        im1b[i]-=thresholdF;
        im1[i]-=thresholdM;
    }
    
	float *warped1=new float[m*n*o];
    
    //READ AFFINE MATRIX from linearBCV if provided (else start from identity)
    
    float* X=new float[16];
    
    if(args.affine){
        size_t split_affine=args.affine_file.find_last_of("/\\");
        if(split_affine==string::npos){
            split_affine=-1;
        }

        cout<<"Reading affine matrix file: "<<args.affine_file.substr(split_affine+1)<<"\n";
        ifstream matfile;
        matfile.open(args.affine_file);
        for(int i=0;i<4;i++){
            string line;
            getline(matfile,line);
            sscanf(line.c_str(),"%f  %f  %f  %f",&X[i],&X[i+4],&X[i+8],&X[i+12]);
        }
        matfile.close();


    }
    else{
        cout<<"Starting with identity transform.\n";
        fill(X,X+16,0.0f);
        X[0]=1.0f; X[1+4]=1.0f; X[2+8]=1.0f; X[3+12]=1.0f;
    }
    
    for(int i=0;i<4;i++){
        printf("%+4.3f | %+4.3f | %+4.3f | %+4.3f \n",X[i],X[i+4],X[i+8],X[i+12]);//X[i],X[i+4],X[i+8],X[i+12]);
        
    }
    
    //PATCH-RADIUS FOR MIND/SSC DESCRIPTORS

    vector<int> mind_step;
    for(int i=0;i<args.quantisation.size();i++){
        mind_step.push_back(floor(0.5f*(float)args.quantisation[i]+1.0f));
    }
    printf("MIND STEPS, %d, %d, %d, %d, %d \n",mind_step[0],mind_step[1],mind_step[2],mind_step[3],mind_step[4]);

	
    
	int step1; int hw1; float quant1;
	
	//set initial flow-fields to 0; i indicates backward (inverse) transform
	//u is in x-direction (2nd dimension), v in y-direction (1st dim) and w in z-direction (3rd dim)
	float* ux=new float[sz]; float* vx=new float[sz]; float* wx=new float[sz];
	for(int i=0;i<sz;i++){
		ux[i]=0.0; vx[i]=0.0; wx[i]=0.0;
	}
	int m2,n2,o2,sz2;
	int m1,n1,o1,sz1;
	m2=m/args.grid_spacing[0]; n2=n/args.grid_spacing[0]; o2=o/args.grid_spacing[0]; sz2=m2*n2*o2;
	float* u1=new float[sz2]; float* v1=new float[sz2]; float* w1=new float[sz2];
	float* u1i=new float[sz2]; float* v1i=new float[sz2]; float* w1i=new float[sz2];
	for(int i=0;i<sz2;i++){		
		u1[i]=0.0; v1[i]=0.0; w1[i]=0.0;
		u1i[i]=0.0; v1i[i]=0.0; w1i[i]=0.0;
	}
	
    float* warped0=new float[m*n*o];
    warpAffine(warped0,im1,im1b,X,ux,vx,wx);
    

    uint64_t* im1_mind=new uint64_t[m*n*o];
    uint64_t* im1b_mind=new uint64_t[m*n*o];
    uint64_t* warped_mind=new uint64_t[m*n*o];
	
	time1a = chrono::high_resolution_clock::now();
    chrono::duration<double> timeDataSmooth;
	//==========================================================================================
	//==========================================================================================

    for(int level=0;level<args.levels;level++){
        quant1=args.quantisation[level];
		
        float prev=mind_step[max(level-1,0)];//max(min(label_quant[max(level-1,0)],2.0f),1.0f);
        float curr=mind_step[level];//max(min(label_quant[level],2.0f),1.0f);
        
        chrono::duration<float> timeMIND, timeSmooth, timeData, timeTrans;
		
        if(level==0|prev!=curr){
            time1 = chrono::high_resolution_clock::now();
            descriptor(im1_mind,warped0,m,n,o,mind_step[level]);//im1 affine
            descriptor(im1b_mind,im1b,m,n,o,mind_step[level]);
            time2 = chrono::high_resolution_clock::now();
            timeMIND = time2 - time1;
		}
		
		step1=args.grid_spacing[level];
		hw1=args.search_radius[level];
		
		int len3=pow(hw1*2+1,3);
		m1=m/step1; n1=n/step1; o1=o/step1; sz1=m1*n1*o1;
        
        float* costall=new float[sz1*len3]; 
        float* u0=new float[sz1]; float* v0=new float[sz1]; float* w0=new float[sz1];
		int* ordered=new int[sz1]; int* parents=new int[sz1]; float* edgemst=new float[sz1];
		
        cout<<"==========================================================\n";
		cout<<"Level "<<level<<" grid="<<step1<<" with sizes: "<<m1<<"x"<<n1<<"x"<<o1<<" hw="<<hw1<<" quant="<<quant1<<"\n";
		cout<<"==========================================================\n";
		
        //FULL-REGISTRATION FORWARDS
        time1 = chrono::high_resolution_clock::now();
		upsampleDeformationsCL(u0,v0,w0,u1,v1,w1,m1,n1,o1,m2,n2,o2);
        upsampleDeformationsCL(ux,vx,wx,u0,v0,w0,m,n,o,m1,n1,o1);
        //float dist=landmarkDistance(ux,vx,wx,m,n,o,distsmm,casenum);
		warpAffine(warped1,im1,im1b,X,ux,vx,wx);
		u1=new float[sz1]; v1=new float[sz1]; w1=new float[sz1];
        time2 = chrono::high_resolution_clock::now();
		timeTrans = time2 - time1;
        cout<<"T"<<flush;
        time1 = chrono::high_resolution_clock::now();
		descriptor(warped_mind,warped1,m,n,o,mind_step[level]);

        time2 = chrono::high_resolution_clock::now();
		timeMIND += time2 - time1;
        cout<<"M"<<flush;
        time1 = chrono::high_resolution_clock::now();
        dataCostCL((unsigned long*)im1b_mind,(unsigned long*)warped_mind,costall,m,n,o,len3,step1,hw1,quant1,args.alpha,RAND_SAMPLES);
        time2 = chrono::high_resolution_clock::now();

		timeData = time2 - time1;
        cout<<"D"<<flush;
        time1 = chrono::high_resolution_clock::now();
        primsGraph(im1b,ordered,parents,edgemst,step1,m,n,o);
        regularisationCL(costall,u0,v0,w0,u1,v1,w1,hw1,step1,quant1,ordered,parents,edgemst);
        time2 = chrono::high_resolution_clock::now();
		timeSmooth = time2 - time1;
        cout<<"S"<<flush;
		
        //FULL-REGISTRATION BACKWARDS
        time1 = chrono::high_resolution_clock::now();
		upsampleDeformationsCL(u0,v0,w0,u1i,v1i,w1i,m1,n1,o1,m2,n2,o2);
        upsampleDeformationsCL(ux,vx,wx,u0,v0,w0,m,n,o,m1,n1,o1);
		warpImageCL(warped1,im1b,warped0,ux,vx,wx);
		u1i=new float[sz1]; v1i=new float[sz1]; w1i=new float[sz1];
        time2 = chrono::high_resolution_clock::now();
		timeTrans += time2 - time1;
        cout<<"T"<<flush;
        time1 = chrono::high_resolution_clock::now();
		descriptor(warped_mind,warped1,m,n,o,mind_step[level]);

        time2 = chrono::high_resolution_clock::now();
		timeMIND += time2 - time1;
        cout<<"M"<<flush;
        time1 = chrono::high_resolution_clock::now();
        dataCostCL((unsigned long*)im1_mind,(unsigned long*)warped_mind,costall,m,n,o,len3,step1,hw1,quant1,args.alpha,RAND_SAMPLES);
        time2 = chrono::high_resolution_clock::now();
		timeData += time2 - time1;
        cout<<"D"<<flush;
        time1 = chrono::high_resolution_clock::now();
        primsGraph(warped0,ordered,parents,edgemst,step1,m,n,o);
        regularisationCL(costall,u0,v0,w0,u1i,v1i,w1i,hw1,step1,quant1,ordered,parents,edgemst);
        time2 = chrono::high_resolution_clock::now();
		timeSmooth += time2 - time1;
        cout<<"S"<<flush;
		
        cout << "\nTime: MIND=" << timeMIND.count() << ", data="<< timeData.count() <<", MST-reg="<< timeSmooth.count() <<", transf.="<< timeTrans.count() <<"\n";
        cout << "speed=" << 2.0*(float)sz1*(float)len3/((timeData+timeSmooth).count()) << " dof/s\n";
        
        time1 = chrono::high_resolution_clock::now();
        consistentMappingCL(u1,v1,w1,u1i,v1i,w1i,m1,n1,o1,step1);
        time2 = chrono::high_resolution_clock::now();
        chrono::duration<float> timeMapping= time2 - time1;
		
        //cout<<"Time consistentMapping: "<<timeMapping<<"  \n";
		
		//upsample deformations from grid-resolution to high-resolution (trilinear=1st-order spline)
		float jac=jacobian(u1,v1,w1,m1,n1,o1,step1);
		
        cout<<"SSD before registration: "<<SSD0<<" and after "<<SSD1<<"\n";
		m2=m1; n2=n1; o2=o1;
		cout<<"\n";
       
		delete u0; delete v0; delete w0;
		delete costall;
        
		delete parents; delete ordered;
        
		
	}
    delete im1_mind;
    delete im1b_mind;
	//==========================================================================================
	//==========================================================================================
	
    time2a = chrono::high_resolution_clock::now();
	chrono::duration<float> timeALL = time2a - time1a;
    
    upsampleDeformationsCL(ux,vx,wx,u1,v1,w1,m,n,o,m1,n1,o1);
	
    float* flow=new float[sz1*3];
	for(int i=0;i<sz1;i++){
		flow[i]=u1[i]; flow[i+sz1]=v1[i]; flow[i+sz1*2]=w1[i];
        //flow[i+sz1*3]=u1i[i]; flow[i+sz1*4]=v1i[i]; flow[i+sz1*5]=w1i[i];
		
	}
    
    //WRITE OUTPUT DISPLACEMENT FIELD AND IMAGE
    writeOutput(flow,outputflow.c_str(),sz1*3);
    warpAffine(warped1,im1,im1b,X,ux,vx,wx);
	
    for(int i=0;i<sz;i++){
        warped1[i]+=thresholdM;
    }
    
    gzWriteNifti(outputfile,warped1,header,m,n,o,1);

    cout<<"SSD before registration: "<<SSD0<<" and after "<<SSD1<<"\n";
    
    // if SEGMENTATION of moving image is provided APPLY SAME TRANSFORM
    if(args.segment){
        short* seg2;
        readNifti(args.moving_seg_file,seg2,header,M,N,O,P);
        
        short* segw=new short[sz];
        fill(segw,segw+sz,0);

        warpAffineS(segw,seg2,X,ux,vx,wx);
        
        
        string outputseg;
        outputseg.append(args.output_stem);
        outputseg.append("_deformed_seg.nii.gz");
        cout<<"outputseg "<<outputseg<<"\n";
        

        
        gzWriteSegment(outputseg,segw,header,m,n,o,1);
    }
    
	cout<<"Finished. Total time: "<< timeALL.count() <<" sec. ("<< timeDataSmooth.count() <<" sec. for MIND+data+reg+trans)\n";
	
	
	return 0;
}
