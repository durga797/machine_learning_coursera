function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

        J = costFunction(theta,X,y);
        x = 0;
        for i = 2:length(theta)
            x = x + (theta(i))^2;
        end
        x = (lambda/(2*m))*x;
        J = J+x;
        z = X*theta;
        h = sigmoid(z);
        grad(1) = sum(h-y)/m;
        for i = 2:length(grad)
            grad(i) = ((h-y)'*X(:,i)+lambda*theta(i))/m;
        end    
        


% =============================================================

end
