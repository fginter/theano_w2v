import sys
import theano
import numpy
import theano.tensor as T

class VSpaceLayer(object):

    """A layer which has a vector of numerical indices on its input
    and the corresponding matrix rows on the output. This is to be
    used as a minibatch. The matrix are the embeddings."""

    @classmethod
    def from_matrix(cls,matrix,input):
        """Input should be an int vector which selects the matrix rows"""
        return cls(matrix.astype(theano.config.floatX),input)

    def __init__(self,matrix,input):
        self.input=input 
        self.wordvecs = theano.shared( #Shared variable for the matrix
            value=matrix,
            name='M',
            borrow=True #don't make own copy
        )
        self.output=self.wordvecs[self.input]
        self.params=[self.wordvecs] #The word vectors should be trained, so they're a parameter



class SoftMaxLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    @classmethod
    def empty(cls,n_in,classes,input=None,rng=None):
        if input is None:
            input=T.matrix('x',theano.config.floatX)
        if rng is None:
            rng = numpy.random.RandomState(5678)
        n_out=len(classes)
        W = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(n_in, n_out)),
                dtype=theano.config.floatX
            )
        b=numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(n_out,)),theano.config.floatX)
        return cls(input,W,b,classes)

    
    def __init__(self, input, W, b, classes):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.classes=classes
        self.W = theano.shared(
            value=W,
            name='W',
            borrow=True
        )
        self.b = theano.shared(value=b,
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k

        self.input=input
        self.p_y_given_x = T.dot(input, self.W) + self.b#T.nnet.softmax(T.dot(input, self.W) + self.b)
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1
        # parameters of the model
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

class SkipGram(object):
    
    """
    Skipgram is really a vector layer on top of which sits the softmax, that's
    sort of all there is to it.
    """
    
    @classmethod
    def empty(cls,inp_vocabulary,outp_vocabulary,dimensionality):

        input=T.ivector('inp') #Symbolic variable for my input
        #Array to store the embeddings
        rng = numpy.random.RandomState(5678)
        vspace_matrix = numpy.asarray(rng.uniform(low=-0.01,high=0.01,size=(len(inp_vocabulary.vocab),dimensionality)),
                dtype=theano.config.floatX
            )

        #vspace_matrix=numpy.asarray(zeros((len(vocabulary.vocab),dimensionality),dtype=theano.config.floatX) #Creates a numpy array, not theano variable
        #..and a vspace layer with those
        vspace_layer=VSpaceLayer.from_matrix(vspace_matrix,input)
        #...now a softmax layer
        softmax_layer=SoftMaxLayer.empty(n_in=dimensionality,classes=outp_vocabulary.vocab,input=vspace_layer.output,rng=None)
        
        return cls(vspace_layer,softmax_layer)
        
        
    def __init__(self,vspace_layer,softmax_layer):
        self.vspace_layer=vspace_layer
        self.softmax_layer=softmax_layer
        self.input=self.vspace_layer.input
        self.params=self.vspace_layer.params #softmax layer(s) are dealt with separately
        self.train=self.compile_train_function(self.softmax_layer) #Call .train(X,Y,l_rate)
        self.outputf=self.compile_output_function(self.softmax_layer) #Call .output(X)


    def compile_train_function(self,softmax_layer):
        """Builds the function self.train_classification(x,y,l_rate) which returns the cost. softmax_layer should be one
        of the softmax_layers in the model (there's only one right now, but could be more)"""

        x = T.ivector('x')  # minibatch, input  --- individual focus words
        y = T.ivector('y')  # minibatch, output --- individual context words to go with them
        l_rate = T.scalar('lrate',theano.config.floatX) #Learning rate

        neg_likelihood=softmax_layer.negative_log_likelihood(y) #The output to optimize
        classification_cost=neg_likelihood ###Maybe add some form of regularization...?

        params=self.params+softmax_layer.params

        gparams = [T.grad(classification_cost, param) for param in params] #Gradient w.r.t. every parameter

        updates = [
            (param, param - l_rate * gparam)
            for param, gparam in zip(params, gparams)
            ]

        # compiling a Theano function `train_model` that returns the cost, but
        # at the same time updates the parameter of the model based on the rules
        # defined in `updates`
        return theano.function(
            inputs=[x,y,l_rate],
            outputs=classification_cost,
            updates=updates,
            givens={
                self.vspace_layer.input: x
                }
            )

    def compile_output_function(self,softmax_layer):
        """Builds the function self.train_classification(x,y,l_rate) which returns the cost. softmax_layer should be one
        of the softmax_layers in the model (there's only one right now, but could be more)"""

        x = T.ivector('x')  # minibatch, input  --- individual focus words
        # compiling a Theano function `train_model` that returns the cost, but
        # at the same time updates the parameter of the model based on the rules
        # defined in `updates`
        return theano.function(
            inputs=[x],
            outputs=softmax_layer.p_y_given_x,
            givens={
                self.vspace_layer.input: x
                }
            )

