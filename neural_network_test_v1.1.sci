clear
mode(0);
rand("normal");
filename = fullfile('C:\Users\Esteban\Documents\SCILAB_ANN\', 'test_data2.csv');
[M,c] = csvRead(filename);
  
InputNeurons=5;
testsize=4;
xhat=zeros(1,testsize);
test_data = [M(2:1026,2:5)'./100];

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

load("C:\Users\Esteban\Documents\SCILAB_ANN\WT7.dat","W1","W2","B1","B2","eta");

    for i=1:testsize
        input_data=test_data(i,:)';
        FLout=afht(W1,B1,input_data);
        SLout=afsig(W2,B2,FLout);
        xhat(1,i)=SLout; 
        if SLout > 0.5 
            mprintf('Epilepsia   '+string(SLout)+'\n')
         else
            mprintf('Normal      '+string(SLout)+'\n')
        end
    end

