clear
mode(0);
rand("normal");
filename = fullfile('C:\Users\Esteban\Documents\SCILAB_ANN\', 'training_data.csv');
[M,c] = csvRead(filename);

// Simulation parameters

/* Training Parameters are the learning rate, which in this case can be 
   selected to be constant or adaptive. The number of training epochs is 
   also defined here. */
eta=0.001;     //Initial learning rate, used only at the begining of training
epochs=1000000;  

InputNeurons=5;
TrainingSetSize=80;
xhat=zeros(1,TrainingSetSize);
e=zeros(1,TrainingSetSize);
esum=zeros(1,epochs);
training_data = [M(2:1026,2:81)'./100];

/* Function prototypes of the Artificial Neural Network forward pass are 
   defined here. Three different activation functions can be defined. 
   Hyperbolic tangent, sigmoid and linear activiation functions. */
//Forward Pass hyperbolic-Tangent Activation Function 
function [y]=afht(w,b,p)
    y=tanh(w*p+b);
endfunction

//Forward Pass Sigmoid Activation Function
function [y]=afsig(w,b,p)
    y=1./(1+exp(-(w*p+b)));
endfunction


/* Function prototpypes of the ANN weight updating rules with 
   backpropagation are defined here. An updating rule is defined for each 
   type of activation function. Only two layers are considered for this 
   ANN algorithm.*/

//Backpropagation for a Sigmoid Activation Function (output layer)
function [wo, bo, deltao]=BPSigmOutputLayer(wo,bo,err,eta,slout,flout)
    deltao=(err.*(slout.*(1-slout)));
    wo=wo+eta*(deltao*flout');
    bo=bo+eta*deltao;
endfunction


//Backpropagation for a Hyperbolic-Tangent Activation Function (hiden layer)
function [wi,bi,deltai]=BPtanhInputLayer(wi,wo,bi,eta,flout,deltao,P)
    deltai=(1-flout.^2).*(wo'*deltao);
    wi=wi+eta*(deltai*P');
    bi=bi+eta*deltai;
endfunction

//Definiton of the ANN wights, inputs and outputs
/* Setting up the ANN parameters like, input delays, number of input 
   neurons, number of output neurons, number of input signals, number 
   output signals, size of weight matrices, and bias matrices. The number
   of layers is fixed to two. */
nin=InputNeurons;       // Number of Input Neurons
non=1;                  // Number of Output Neurons
nbi=nin;                // Number of Input bias
nbo=non;                // Number of Output bias
nis=1025;                // Number of Input Signals or Data
nos=nin;                // Number of Input Signals to the Output Layer


//ANN initial weights
W1=0.5*rand(nin,nis);
W2=0.5*rand(non,nos);
B1=0.5*rand(nin,1);
B2=0.5*rand(non,1);

//Load weights and learning rate from a file
//load("C:\Users\Esteban\Documents\SCILAB_ANN\WT.dat","W1","W2","B1","B2","eta");
//eta=0.05;

// Simulation
for k=1:epochs
    for i=1:TrainingSetSize
        input_data=training_data(i,:)';
        if i <= 40
        FLout=afht(W1,B1,input_data);
        SLout=afsig(W2,B2,FLout);
        xhat(1,i)=SLout;            // ANN model output
        e(1,i)=0.9-xhat(1,i); 
    else
        FLout=afht(W1,B1,input_data);
        SLout=afsig(W2,B2,FLout);
        xhat(1,i)=SLout;            // ANN model output
        e(1,i)=0.2-xhat(1,i); 
    end
             // modeling error
        
//  Backward pass - Back Propagation
     [W2,B2,deltao]=BPSigmOutputLayer(W2,B2,e(:,i),eta,SLout,FLout);
     [W1,B1,deltai]=BPtanhInputLayer(W1,W2,B1,eta,FLout,deltao,input_data);
        
    end    
    esum(1,k)=(norm(e(1,:)/TrainingSetSize));
    
//  Sum of the error
//Without adaptive upgrading of the learning rate    
    mprintf('epoch=%i, Training error=%f\n',k,esum(1,k))
    if esum(1,k)<=0.01    //Tasa de error 
        break
    end
    esum(1,k)=0;
end

mprintf('Saving Weights and eta\n')
save("C:\Users\Esteban\Documents\SCILAB_ANN\WT7.dat","W1","W2","B1","B2","eta");
mprintf('Done!')
