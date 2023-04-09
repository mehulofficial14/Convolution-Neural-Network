# Convolutional Neural Networks

- Architecture and operations: Image-related tasks(similar to the visual cortex in the brain)
- Can not use a normal neural network: a large number of parameters(each pixel in the large image will have a certain weight based on which the loss is calculated)
- Convolutional Neural Networks: partially connected layers and weight sharing
- Mainly 2 types of layers:
    - Convolutional Layers
    - Pooling Layers
- Convolutional Layer:
    - Set of filters called kernels: define the modification of the input
    - Kernel for vertical edges
        
        $$
        \begin{bmatrix}
            1&0&-1\\
            1&0&-1\\
            1&0&-1 
        \end{bmatrix}
        $$
        
    - Kernel for horizontal edges
        
        $$
        \begin{bmatrix}
            1&1&1\\
            0&0&0\\
            -1&-1&-1 
        \end{bmatrix}
        $$
        
    - Kernel values: learned in the training process
    - Convolution isolates different features that are then used in the dense layer
- Pooling Layer:
    - Shrink input: reduce the computational load and memory consumption(reduce parameters)
    - The aggregate portion of the output of the convolutional layer to a single value
- Code:
    - Create a class, ConvolutionLayer with a constructor that takes in the inputs: number of kernels and the size of the kernel(assuming a square kernel)
        
        ```python
        class ConvolutionLayer:
            def __init__(self, kernel_num, kernel_size):
                self.kernel_num = kernel_num
                self.kernel_size = kernel_size        
                self.kernels = np.random.normal(loc=0, scale=np.sqrt(2 / (kernel_size ** 2 + kernel_num)), size=(kernel_num, kernel_size, kernel_size))
        ```
        
        The kernels are randomly defined but to make sure that the kernels do not increase or decrease in an exponential manner, the normalization of weights is done using Xavier’s initialization.
        
        ```python
        self.kernels = np.random.normal(loc=0, scale=np.sqrt(2 / (kernel_size ** 2 + kernel_num)), size=(kernel_num, kernel_size, kernel_size))
        ```
        
        Xavier’s initialization is as follows:
        
        - If the number of inputs of a layer is n and the number of outputs from a layer is m then the weights should follow the normal distribution: $X \sim N(\sigma, \mu)$ where $\mu = 0$ and $\sigma = \sqrt \frac {2} {n + m}$
        - The number of inputs(n) is proportional to the total units in a kernel or the number of input parameters or weights for the convolutional layer. The number of outputs(m) is equal to the number of kernels as each kernel gives an output per number of kernels.
    - Create a function, patch_generator that takes in the image as an input and processes patches of the image that are the same size as that of the kernel.
        
        ```python
        		def patches_generator(self, image):
                image_h, image_w = image.shape
                self.image = image
                for h in range(image_h-self.kernel_size+1):
                    for w in range(image_w-self.kernel_size+1):
                        patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
                        yield patch, h, w
        ```
        
        1. Get the image dimensions using the shape attribute of the Numpy object, image.
            
            ```python
            image_h, image_w = image.shape
            ```
            
        2. Iterate over the image and get patches of the dimensions same as that of the kernel size.
            
            ```python
            for h in range(image_h-self.kernel_size+1):
                for w in range(image_w-self.kernel_size+1):
                    patch = image[h:(h+self.kernel_size), w:(w+self.kernel_size)]
            ```
            
            For example, let us consider that the image is of dimensions, 4 pixels height-wise, and 4 pixels width-wise. The kernel is of the dimensions 2 pixels height-wise and 2 pixels width-wise.
            
            |  | 0 | 1 | 2 | 3 |
            | --- | --- | --- | --- | --- |
            | 0 | First Patch | First Patch |  |  |
            | 1 | First Patch | First Patch |  |  |
            | 2 |  |  | Last Patch | Last Patch |
            | 3 |  |  | Last Patch | Last Patch |
            
            The kernel is picked from its topmost corner(i.e. (0, 0) for the kernel starts from (0, 0) of the image and when it ends, it can go up to (2,2) of the image. That is the right-bottommost patch.
            
        3. In each iteration, give the patch and the coordinates of the topmost corner of the kernel with respect to the image to the function that calls the patches_generator function.
            
            ```python
            yield patch, h, w
            ```
            
    - Create a forward propagating function that takes in an image as input and does the forward propagation.
        
        ```python
        def forward_prop(self, image):
              image_h, image_w = image.shape
              convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
              for patch, h, w in self.patches_generator(image):
                  convolution_output[h,w] = np.sum(patch*self.kernels, axis=(1,2))
              return convolution_output
        ```
        
        Considering the previous example, the input is a 4 by 4 matrix and the kernel is of size 2 by 2, and a convolutional operation multiplies the patch with the kernel and adds the values.
        
        Suppose the input image is this.
        
        | 1 | -2 | -9 | 6 |
        | --- | --- | --- | --- |
        | 5 | 8 | 7 | 9 |
        | -6 | 2 | 0 | -1 |
        | -3 | 4 | -1 | 0 |
        
        The kernel is this.
        
        | 6 | 0 |
        | --- | --- |
        | -7 | -1 |
        
        So the output for the first patch looks like this.
        
        $(6 * 1) + (0 * -2) + (-7 * 5) + (-1 * 8) = -37$
        
        The final output looks like this.
        
        | -37 | -75 | -112 |
        | --- | --- | --- |
        | 70 | 34 | 43 |
        | -19 | -15 | 7 |
        
        If the input image is of the shape, $(I_h, I_w)$ and the kernel is of the shape, $(K_h, K_w)$ then the output is of the shape, $(I_h - K_h + 1, I_w - K_w + 1)$ for each kernel output. This can be seen in this code.
        
        ```python
        convolution_output = np.zeros((image_h-self.kernel_size+1, image_w-self.kernel_size+1, self.kernel_num))
        ```
        
        The output of one convolutional operation can be defined as $Y_{(h, w)} = \sum_{i = h}^{h + K_h - 1} ~ \sum_{j = w}^{w + K_w - 1} I_{(i, j)} ~ * ~ K_{(i - h, j - w)} ~ \forall ~ 0 \le h \le (I_h - K_h + 1) ~ and ~ 0 \le w \le (I_w - K_w + 1)$ where the “*” operation is an element-wise product and Y is the output matrix of the convolutional layer.
        
    - Create a backward propagation that takes in the gradient of the error/loss of the whole neural network(E) with respect to the output of the convolutional layer(Y). It also takes in the learning rate.
        
        ```python
        def back_prop(self, dE_dY, alpha):
            dE_dk = np.zeros(self.kernels.shape)
            for patch, h, w in self.patches_generator(self.image):
                for f in range(self.kernel_num):
                    dE_dk[f] += patch * dE_dY[h, w, f]
            self.kernels -= alpha*dE_dk
            return dE_dk
        ```
        
        The predicted value is the result of the forward propagation. For a patch, the predicted value of a patch is this.
        
        $Y_{pred_{patch}} = X_{patch} * K$
        
        The trainable parameters here are K. To update K, the derivative of the final loss function with respect to the current output is required along with the learning rate.
        
        $\frac {\partial{E}} {\partial{K}} = \frac {\partial{E}} {\partial{Y_{pred_{patch}}}} ~ * ~ \frac {\partial{Y_{pred_{patch}}}} {\partial{K}}$
        
        $K := K - \alpha * \frac {\partial{E}} {\partial{K}}$
        
        $\frac {\partial{Y_{pred_{patch}}}} {\partial{K}} = X_{patch} * I$
        
        The above happens for every kernel and can be done using a loop. The input(X) is constant and $\frac{\partial {K}}{\partial {K}} = I$.
        
    - Create a MaxPooling class with a constructor that takes in the kernel size.
        
        ```python
        class MaxPoolingLayer:
            def __init__(self, kernel_size):
                self.kernel_size = kernel_size
        ```
        
    - Get the patches using the patches_generator. The maximum is found in every patch. The dimensions of the input are $I_w \times I_h \times C$ where C is the number of kernels. The size of the output width and output height is the number of time a kernel can move horizontally and vertically.
        
        ```python
        def patches_generator(self, image):
            output_h = image.shape[0] // self.kernel_size
            output_w = image.shape[1] // self.kernel_size
            self.image = image
        
            for h in range(output_h):
                for w in range(output_w):
                    patch = image[(h*self.kernel_size):(h*self.kernel_size+self.kernel_size), (w*self.kernel_size):(w*self.kernel_size+self.kernel_size)]
                    yield patch, h, w
        ```
        
    - To get the forward propagation, the function gets the maximum value of each patch. $Y_{(h, w)} = max(X(h + i, w + j)) ~ ~ \forall ~ ~ 0 \le h \le (\lfloor\frac{I_h}{K_h}\rfloor) ~ ~ and ~ ~ 0 \le w \le (\lfloor\frac{I_w}{K_w}\rfloor) ~ ~ and ~ ~ 0 \le i \le K_h ~ ~ and ~ ~ 0 \le j \le K_w$
        
        ```python
        def forward_prop(self, image):
            image_h, image_w, num_kernels = image.shape
            max_pooling_output = np.zeros((image_h//self.kernel_size, image_w//self.kernel_size, num_kernels))
            for patch, h, w in self.patches_generator(image):
                max_pooling_output[h,w] = np.amax(patch, axis=(0,1))
            return max_pooling_output
        ```
        
    - For backward propagation, use the following.
        
        $\frac{\partial E}{ \partial K} = \frac{\partial E}{\partial Y} * \frac{\partial Y}{\partial K}$ where $Y = max(K)$ so $\frac{\partial Y} {\partial K} = 1$ where $Y == max(K)$ else $\frac {\partial Y} {\partial K} = 0$
        
        ```python
        def back_prop(self, dE_dY):
            dE_dk = np.zeros(self.image.shape)
            for patch,h,w in self.patches_generator(self.image):
                image_h, image_w, num_kernels = patch.shape
                max_val = np.amax(patch, axis=(0,1))
        
                for idx_h in range(image_h):
                    for idx_w in range(image_w):
                        for idx_k in range(num_kernels):
                            if patch[idx_h,idx_w,idx_k] == max_val[idx_k]:
                                dE_dk[h*self.kernel_size+idx_h, w*self.kernel_size+idx_w, idx_k] = dE_dY[h,w,idx_k]
            return dE_dk
        ```
        
    - There are no weights to be updated in MaxPooling layers.
    - Create a SoftmaxLayer class with a constructor that takes in the number of inputs and outputs. Use Xavier’s initialization to normalize the weights.
        
        ```python
        class SoftmaxLayer:
            def __init__(self, input_units, output_units):
                stddev = np.sqrt(2 / (input_units + output_units))
                mean = 0
                self.weight = np.random.normal(loc=mean, scale=stddev, size=(input_units, output_units))
                self.bias = np.zeros(output_units)
        ```
        
    - For forward propagation, first flatten the input and then use weights to create this.
        
        $Softmax(z_i) = \frac{e^{z_i}}{\sum e^{z_i}}$
        
        ```python
        def forward_prop(self, image):
            self.original_shape = image.shape
            image_flattened = image.flatten()
            self.flattened_input = image_flattened
            first_output = np.dot(image_flattened, self.weight) + self.bias
            self.output = first_output
            softmax_output = np.exp(first_output) / np.sum(np.exp(first_output), axis=0)
            return softmax_output
        ```
        
    - For the backward propagation, use this.
        
        ```python
        def back_prop(self, dE_dY, alpha):
            for i, gradient in enumerate(dE_dY):
                if gradient == 0:
                    continue
                transformation_eq = np.exp(self.output)
                S_total = np.sum(transformation_eq)
        
                dY_dZ = -transformation_eq[i]*transformation_eq / (S_total**2)
                dY_dZ[i] = transformation_eq[i]*(S_total - transformation_eq[i]) / (S_total**2)
        
                dZ_dw = self.flattened_input
                dZ_db = 1
                dZ_dX = self.weight
        
                dE_dZ = gradient * dY_dZ
        
                dE_dw = dZ_dw[np.newaxis].T @ dE_dZ[np.newaxis]
                dE_db = dE_dZ * dZ_db
                dE_dX = dZ_dX @ dE_dZ
        
                self.weight -= alpha*dE_dw
                self.bias -= alpha*dE_db
        
            return dE_dX.reshape(self.original_shape)
        ```