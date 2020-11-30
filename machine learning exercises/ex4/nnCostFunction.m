function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25*401
Theta2_grad = zeros(size(Theta2)); % 10*26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Forward propagation
% 5000 training examples, 400 pixels/input layer units
% First layer
a1 = [ones(m,1), X]; % 5000*401

% Second layer
z2 = a1 * Theta1'; % 5000*25, 25 units in second layer
a2 = [ones(length(z2),1), sigmoid(z2)]; % 5000*26

% Third layer (output layer)
z3 = a2 * Theta2'; % 5000*10, 10 units output layer
a3 = sigmoid(z3); % 5000*10

% Hypothesis
h = a3; % 5000*10

% build matrix Y out of rows indexed from the identity matrix, using a single indexing operation
I = eye(num_labels);
Y = I(y,:); % 5000*10

% size(Theta1_rg1)=25*401
Theta_rg1 = [zeros(size(Theta1,1),1),Theta1(:, 2:end)];
% size(Theta2_rg2)=10*26
Theta_rg2 = [zeros(size(Theta2,1),1),Theta2(:, 2:end)];

J = (-1/m)*sum((Y.*log(h)+(1-Y).*log(1-h)), 'all')+(lambda/(2*m))*(sum(Theta_rg1.^2, 'all')+sum(Theta_rg2.^2, 'all'));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d3 = a3-Y; % 5000*10
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); % 5000*25

Delta1 = d2' * a1; % 25*401
Delta2 = d3' * a2; % 10*26

Theta1_grad = Delta1/m + (lambda/m)*Theta_rg1; % 25*401
Theta2_grad = Delta2/m + (lambda/m)*Theta_rg2; % 10*26

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
