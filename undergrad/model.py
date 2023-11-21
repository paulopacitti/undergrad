import numpy as np

class Model:
    def __init__(self, layers_dims: List[int],
                 activation_funcs: List[BaseFunction],
                 initialization_method: str = "random"):
        """
        Argumentos:
        layers_dims: (list) lista com o tamanho de cada camada
        activation_funcs: (list) lista com as funções de ativação
        initialization_method: (str) indica como inicializar os parâmetros

        Exemplo:

        # Um modelo de arquitetura com camadas 2 x 1 x 2 e 2 ReLU como funções de ativação
        >>> m = Model([2, 1, 2], [ReLU(), ReLU()])
        """

        assert all([isinstance(d, int) for d in layers_dims]), \
        "É esperado uma lista de int como o parâmetro ``layers_dims"

        assert all([isinstance(a, BaseFunction) for a in activation_funcs]), \
        "É esperado uma lista de BaseFunction como o parâmetro ``activation_funcs´´"

        self.layers_dims = layers_dims
        self.activation_funcs = activation_funcs
        self.weights, self.bias = self.initialize_model(initialization_method)


    def __len__(self):
        return len(self.weights)


    def initialize_model(self, method="random"):
        """
        Argumentos:
        layers_dims: (list)  lista com o tamanho de cada camada
        method: (str) indica como inicializar os parâmetros

        Retorna: uma lista de matrizes (np.array) de pesos e
        uma lista de matrizes (np.array) como biases.
        """

        weights = []
        bias = []
        n_layers = len(self.layers_dims)
        for l in range(0, n_layers-1):
            # o peso w_i,j  conecta o i-th neurônio na camada atual para
            # o j-th neurônio na próxima camada
            W = np.random.randn(self.layers_dims[l], self.layers_dims[l + 1])
            b = np.random.randn(1, self.layers_dims[l + 1])

            # He et al. Inicialização Normal
            if method.lower() == 'he':
                W = W * np.sqrt(2/self.layers_dims[l])
                b = b * np.sqrt(2/self.layers_dims[l])
            # TODO: implemente outro método de inicialização
            if method.lower() == 'xavier':
                W = W * np.sqrt(1/np.mean([self.layers_dims[l], self.layers_dims[l+1]]))
                b = b * np.sqrt(1/np.mean([self.layers_dims[l], self.layers_dims[l+1]]))

            weights.append(W.astype(np.float64))
            bias.append(b.astype(np.float64))

        return weights, bias


    def forward(self, X):
        """
        Argumentos:
        X: (np.array) dados de entrada

        Retorno:
        Predições para os dados de entrada (np.array)
        """
        activation = X
        self.activations = [X]
        self.Z_list = []
        #############################################################################
        # TODO: implemente aqui o forward step.
        #
        # Mais algumas instruções:
        # Note que os pesos, bias e funções de ativações são variáveis de classe,
        # então você pode acessá-los através do ``self``.
        #
        # Você deve armazenar a entrada Z de cada função de ativação em ``Z_list``,
        # e a saída das funções de ativação em ``ativações``. Esses
        # informações serão importantes quando você implementar a passagem para trás.
        #############################################################################
        for layer in range(len(self.weights)):
            z = np.dot(activation, self.weights[layer]) + self.bias[layer]
            self.Z_list.append(z)
            activation = self.activation_funcs[layer](z)
            self.activations.append(activation)
        return self.activations[-1]
