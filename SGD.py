import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import streamlit as st
import time



time.sleep(0.5)

st.set_page_config(
    page_title="Descida do Gradient",
    layout="wide",
    )

fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
class Map(object):
    
        def __init__(self, theta, noise_s, n_points, lmbda1, lmbda2):
            np.random.seed(5)
            self.theta = theta
            self.X  = np.array(np.random.randn(n_points,2))
            self.Z  = self.f(theta,self.X)          
            self.X2 = self.X + noise_s*np.random.randn(n_points, 2)
            self.theta_opt = self.Opt()   
            self.noise_s = noise_s
            self.dados = np.vstack((self.X2.T,self.Z)).T

            self.l1 = lmbda1
            self.l2 = lmbda2
                
        @staticmethod
        def f(Teta,Xis):
            return np.dot(Xis, Teta)
        
        def Opt(self):
            a1 = np.dot(self.X2.T, self.X2)
            a2 = np.dot(self.X2.T, self.Z )
            Theta_opt = np.dot( np.linalg.inv( a1 ), a2 ) 
            return Theta_opt
        
    
        def plot_map(self):
            t = np.linspace(-10,10,50) 
            L = np.zeros((t.size,t.size))
            for i,t1 in enumerate(t):
                for j,t2 in enumerate(t):
                    T = np.array([t2,t1])
                    L[i,j] = ((self.f(T,self.X2) - self.Z)**2).mean() + self.l1*np.sum(np.abs(T)) + self.l2*np.dot(T.T,T)

            ax.contour( t,t,L, levels=30,colors='w',linewidths=0.1)
            ax.contourf( t ,t, L, levels=100, cmap='RdBu' )
            ax.plot([0],[0],'w+',ms=7,label='Origin')
            ax.plot( self.theta_opt[0],self.theta_opt[1],'X',color='gray', ms=7, label=r"$\Theta$'"   )
            ax.plot( [self.theta[0]],[self.theta[1]],'g*'    , ms=7, label=r"Ideal $\Theta$"   )
            #ax.set_title('Loss Function Map',fontsize=12)
            ax.set_ylabel(r'$\theta_2$',fontsize=9)
            ax.set_xlabel(r'$\theta_1$',fontsize=9)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.set_xticks(np.arange(-10,12,2))
            ax.set_yticks(np.arange(-10,12,2))
            ax.legend(loc=0,framealpha=0.6,prop={'size': 7})
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            ax.grid(alpha=0.2,which='both') 
               

           
# Classe Regression
class Regression(object):

    def __init__(self, i_theta, learning_rate, epochs, mini_batch_size, lmbda1, lmbda2, semente):
        self.eta    = learning_rate
        self.epochs = epochs
        self.mbs    = mini_batch_size
        self.l1     = lmbda1
        self.l2     = lmbda2
        self.theta  = i_theta
        self.beta   = [self.theta]
        self.rmse   = []      
      

    def SGD(self, dados):
        '''
        Fatos
        '''
    
        n = dados.shape[0]

        for j in range(self.epochs):
            np.random.seed(semente)
            np.random.shuffle( dados )
            mini_batches = [dados[k:k+self.mbs,:] for k in np.arange(0, n, self.mbs)]

            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, n )
                rms = self.cost(mini_batch)
                self.rmse.append(rms)
        #return np.array(self.beta)

    def update_mini_batch(self, mini_batch, n ):
        """Atualiza os pesos e bias da rede aplicando
         a descida do gradiente usando backpropagation para um único mini lote.
         O `mini_batch` é uma lista de tuplas `(x, y)`, e `eta` é a taxa de aprendizado."""

        nabla_theta = [ np.zeros(t.shape) for t in self.theta  ]
        
        for i in enumerate(mini_batch):
            x = mini_batch[:,:2]
            y = mini_batch[:,2]
            delta_nabla_theta = self.dLdt(x, y)
            nabla_theta = [nt+dnt for nt, dnt in zip(nabla_theta, delta_nabla_theta)]

        self.theta = np.array([t - (self.eta/self.mbs)*nt for t, nt in zip(self.theta, nabla_theta)])

        #[print((self.eta/self.mbs)*nt) for t, nt in zip(self.theta, nabla_theta)]
        self.beta.append(self.theta)
  
        
    def dLdt(self, x, y):
        T = np.array(self.theta)
        y_hat = np.dot(T, x.T) 
        
        sinal = np.sign(T)
        #sinal[ sinal == 0] = 1 

        nabla_theta = 2*np.dot( y_hat - y, x )/len(y) + self.l1*sinal  + 2*self.l2*T                                 

        return (nabla_theta)    
    
    def cost(self, dados):
        T = self.theta
        x = dados[:,:2]
        y = dados[:,2]
        
        y_hat = np.dot(x, T.T)                                            
        return np.sum((y_hat - y)**2.0)/len(y) + self.l1*np.sum(np.abs(T)) + self.l2*np.dot(T.T,T)

    
    def plot_sol(self):
        Thetas = np.array(self.beta) #[:,0]
        ax.plot(Thetas[:,0],Thetas[:,1],'-',color='w',lw=1)
        ax.plot(Thetas[:,0][-1],Thetas[:,1][-1],'rD',ms=5,label='Last Solution')
        

    def plot_rmse(self):
        ax2.loglog(np.arange(1,len(self.rmse)+1),self.rmse,'k-',lw=0.3)
        ax2.set_ylabel(r'Loss Function',fontsize=9)
        ax2.set_xlabel(r'Iteration',fontsize=9)
        ax2.tick_params(axis='both', which='major', labelsize=7)
        ax2.grid()
        
        #ax2.set_xticks(np.arange(-10,12,1))
        #ax2.set_yticks(np.arange(-10,12,1))

       
Plot = False
with st.sidebar:
    st.write("## Parâmetros de Entrada")

    st.write("### Semente Aleatória")    
    seed = st.radio('Semente Fixa?', [ "Yes", "No"],horizontal=True) 
    if seed == 'Yes' :
        semente = 123
    else:
        semente = np.random.randint(9999)
        
  
    st.write("### Mapa de Função Perda")
    st.write("##### O problema: y = X$\\Theta$, uma regressão linear em que $\\Theta$ = [$\\theta_1$,$\\theta_2$] ")
    
    npairs = st.number_input('Np: número de amostras', 2, 10000, 100, step = 1)
    theta1 = st.number_input(r'$\theta_1$:', -10., 10., 3., step = 0.1)
    theta2 = st.number_input(r'$\theta_2$:', -10., 10., 3., step = 0.1)

    st.write("##### Adicionando Ruído as Np amostras: :\ny = X'$\\Theta$,  $x_{ij}$' = $x_{ij}$ + $\sigma$N(0,1) ")
    noise  = st.number_input(r'$\sigma$:', 0.0, 10.0, 0.01, step = 0.01)

    st.write("### Chute Inicial ")
    i_t1 = st.number_input(r'$\theta_1(0)$:', -10.0, 10.0, +7.5, step = 0.1)
    i_t2 = st.number_input(r'$\theta_2(0)$:', -10.0, 10.0, -7.5, step = 0.1)
    
    
    st.write("### Parâmetros para a Desicda do Gradiente")
    learning_rate   = st.number_input('Taxa de Aprendizagem:', 0.0, 1.0, 0.05)
    mini_batch_size = st.number_input('Tamanho do Lote:', 1, 1000, 4)
    epochs          = st.number_input('Épocas:', 0, 500, 10)
    

    st.write("### Parâmetros de Regularização")
    l1 = st.number_input(r'LASSO or L1, $\lambda_1$:', -100.000, 100.000, 0.000, step=0.1)
    l2 = st.number_input(r'Ridge or L2, $\lambda_2$:', -100.000, 100.000, 0.000, step=0.1)

    
    #if st.button("Plot", type="primary"):
    Plot = True


#st.title("Stochastic and Mini Batch Gradient Descent")  
st.write("""
         ### Descida do Gradiente 
         ##### uma interpretação visual da técnica, aqui aplicada a um modelo de regressão linear. 
         """)


tab1, tab3, tab2 = st.tabs(["Visualização do Método","Entendendo os Gráficos" ,"Uma Breve Explicação"])
with tab1:

    col1,col2 = st.columns([5,3])
    with col1:
        st.write("### Gráficos ")
        
        st.write(r"##### Mapa da Função Perda, $L(\theta_1,\theta_2)$")
        Mapa = Map( [theta1,theta2], noise, npairs, l1, l2)
        dados = Mapa.dados

        i_theta = np.array([i_t1, i_t2])
        Sol = Regression(i_theta, learning_rate, epochs, mini_batch_size, l1, l2,semente)
        Sol.SGD(dados)
        Sol.plot_sol()

        Mapa.plot_map()
        fig.tight_layout()
        st.pyplot(fig,dpi=300,use_container_width=True)


        st.write("##### Evolução da Função Perda  ")    
        Sol.plot_rmse()
        fig2.tight_layout()
        ax2.set_xlim(1, int(epochs*npairs/mini_batch_size) )
        st.pyplot(fig2,dpi=300,use_container_width=True)

    with col2:
        st.write(r"""
        ##### Testes Sugeridos 
        1. testar os valores $\sigma$ = 0.0, 0.5 e 1.0 para notar que ruídos distanciam as soluções MMQ e DG da solução ótima, levam ambas as soluções
           para mais próximo da origem e adicionam dispersão às últimas iterações da solução DG.    
        2. fixando $\sigma$ = 1.0, 'Tamanho do Lote' = 1 e removendo o 'Fixed Seed',  entrar com 1 no campo 'Tamanho do Lote' várias vezes para observar como ao 
           fim de cada cálculo a solução DG é dispersa, ou aleatória, em torno da solução MMQ (mantida), um comportamento típico de regularizações por 
           evitar o sobreajuste. 
        3. aceitar o 'Semente Fixa', fixar $\sigma$ = 1.0 e testar os 'Tamanho do Lote' = 1, 2, 10 e 100 para notar que o aumento do 'Tamanho do Lote' suaviza a 
           trajetória da DG, que vai em direção à sol MMQ. A suavização ocorre devido à média aplicada na equação 5 da 3ª aba. 
        4. com $\sigma$ de 1.0 testar 'Taxa de Aprendizagem' de 0.01, 0.05 e 0.1 para observar como uma taxa de aprendizagem alta impede a a convergência do GD,
           implicando, também, em um comportamento errático. Vemos mais uma vez o comportamento de regularização. Entende-see que um 'Learning Rate'
           muito pequeno pode levar à especialização do modelo aos dados de treino, ou sobreajuste. 
        5. reinicie os parâmetros (ou página), e teste $\lambda_1$ com 1, 5 e 10 e 50. Embora exagerados, podemos observar três coisas: 1 - a regularização 
           tende a levar a solução DG para  mais próximo da origem quanto maior for $\lambda_1$; 2 - novamente o a solução DG se tende tornar cada vez 
           mais errática com o aumento do parâmetro; 3 - a regularização L1 altera profundamente o mapa da função perda, tendendo a um mapa do tipo 
           $|\theta_1| + |\theta_2| = |constante|$ , indicando o domínio da regularização L1 sobre a função perda.
        6. reinicie os parâmetros , e teste $\lambda_2$ com 1, 5 e 10. Bem como a regularização L1, a solução com L2 move-se em direção à origem enquanto
           aumenta a sua dispersão. Ainda, vê-se que a função perda vai se transformando em um mapa que destaca a superfície $\theta_1^2 + \theta_2^2 = 
           constante^2$.
        7. aumente o número de épocas para observar o aumento no número de iterações. A solução converge? Olhar para o gráfico da Função Perda $\times$ 
           Número de Iteração      
                 
        Embora tenhamos noslimitado em observar as mudanças no mapa da função perda, conclusões similares podem ainda ser obtididas a partir
        do gráfico de Função Perda $\times$ Número de Iteração..
        
        - MMQ: Método dos Mínimos Quadrados
        - DG: Descida do Gradiente      
        """)


with tab2:
    st.write(r'''
    ##### Objetivo
    Esse código tem o intuito de expor visualmente como o método de descida do gradiente alcança uma
    solução 'ótima' para um problema de otimização. Pela facilidade, o problema aqui proposto é uma simples regressão linear
    com duas variáveis explicativas e uma explicada. Este é um problema que possui rápida solução e que por apresentar dois coeficientes
    a serem determinados, permite que observemos a evolução do par 'ótimo' em um mapa 2D. São adicionados também os métodos LASSO 
    e Ridge de regularização para que possamos observar como tais abordagens afetam a solução 'ótima'. 
    

    ##### O Problema
    Vamos supor que tenhamos em mãos um conjunto de dados contendo três variáveis: uma variável dependente, $y$, e duas
    váriáveis independentes, $x_1$ e $x_2$, que se relacionam através de uma simples relação linear: $y = x_1\theta_1 + x_2\theta_2$. Aqui,
    deseja-se dentificar os coeficientes $\theta_1$ e  $\theta_2$. Em notação matricial, o modelo fica:
             
    $y_{_{n \times 1}} = X_{_{n \times 2}}\Theta_{_{2 \times 1}}$ (equação 1)
             
    Se faz importante estabelecer que $\Theta$ muito provavelmente não irá satisfazer a equação acima com total precisão. Dessa forma, uma estimativa
    para $\Theta$ deverá ser aquela que mais 'satisfaz' os conjuto de dados.      
             
    Suponha que tenhamos em nosso banco de dados $n\ge 2$ registros para as três variáveis. Garantido que não haja constante 
    $\alpha$ tal que $x_1 = \alpha x_2$, há quantidade o suficiente de dados para se estimar $\Theta$, que chamaremos de 
    $\hat{\Theta}$. A dintinção entre $\hat{\Theta}$ e $\Theta$ ocorre por ser a primeira uma estimativa da segunda. Se 
    $n=2$, e $det(X) \neq 0$, podemos isolar $\Theta$ na equação 1 multiplicando, à esquerda, ambos os seus lados pela inversa de $X$, chegando à
    equação abaixo:
    
    $\hat{\Theta} = X^{-1}y$, tal que $n=2$. 
    
    ###### Solução Comum 
             
    Infelizmente, a equação acima só é válida quando $X$ é uma matriz quadrada. É comum lidarmos com sistemas em que temos mais dados 
    do que variáveis (em nosso problema, quado $n>2$) e, neste caso, $X$ não adimite a inversa convencional.
    
    Porém, podemos chegar a algo bem parecido através de algumas manipulações: multiplicando à esquerda ambos os lados da equação 1 pela transposta de $X$, 
    chegamos a $X^{T}y = (X^{T}X)\hat{\Theta}$. Note que agora temos, à direita, a matriz $(X^{T}X)$, uma matriz de 
    dimensão 2x2, simétrica e quadrada, e que admite inversa devido à restrição imposta aos dados em $x_1 \neq \alpha x_2$ . Agora, 
    temos condição para isolar  $\hat{\Theta}$ :
             
    $\hat{\Theta} = X^{+}y$ (equação 2) 
    
    válido para $n \ge 2$ e tal que $X^{+}$ é a inversa de Moore-Penrose e é definida por  $X^{+} = (X^{T}X)^{-1}X^{T} $.      

    A partir desse resultado, façamos algumas observações:
             
    1.   $\hat{\Theta}$ é 'apenas' uma estimativa para $\Theta$. Como consequência, o valor da variável dependente calculado a partir 
         de $X\hat{\Theta}$ não será exatamente $y$, mas uma estimativa para tal, denominada $\hat{y}$. Dessa forma, afirma-se que
         $\hat{y} = X\hat{\Theta}$. E qual a relação entre $\hat{y}$ e $y$ ? Utilizando a eq.2,
         chegamos a:
       
         $\hat{y} = X(X^{T}X)^{-1}X^{T}y,$ 
        
         e também a $y = X\hat{\Theta} + \epsilon$, e a $\hat{y} = y - \epsilon$
             
         Geometricamente falando, $\hat{y}$ é a projeção de $y$ no espaço definido pelas colunas de $X$. Se por obra do acaso $y$ (original)
         for perfeitamente definido no espaço das colunas de $X$, temos que $y$ será uma combinação linear dessas colunas, nenhum erro $\epsilon$
         será necessário, e $\hat{y}$ será igual a $y$. Veja que somente $X$ passa intactamente aos cálculos e isso mostra
         o quanto confiamos nas variáveis independentes. Para tanto, [$X$ deve satisfazer a alguns critérios](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/).
           
     
    2.  $\hat{\Theta}$ obtido através da eq. 2 é a melhor estimativa para $\Theta$ que poderíamos encontrar caso seja de nosso interesse
        minimizar $\Sigma_{i=1}^{n}(y-\hat{y})^2$ ou $\Sigma_{i=1}^{n}\epsilon_i^2$, o que corresponde ao método de mínimos quadrados (MMQ).


    ###### Adicionando erros aleatórios a X
    Vamos supor que confiamos em nossas medidas de $y$, mas que erros de origem aleatória afetaram medidas de $X$. Neste caso, 
    qual queria a relação entre as estimativas para $\Theta$ para os casos com e sem erro? Similarmente ao proposto pele problema anterior,
    $y =X\Theta = (X+E)\Theta_E $ e portanto
                   
    $\hat{\Theta}_E = [I - ((X+E)^{T}(X+E))^{-1}(X+E)^{T}E]\Theta$ (equação 3)     

    Aqui temos alguns resultados interessantes: 
    
     - se $E = 0_{n\times 2}$, $\hat{\Theta}_E = \Theta$, como era de se esperar;       
     - se $E = X$, $\hat{\Theta}_E = \Theta/2$ indicando que se $X$ dobra,  $\Theta$ tem que ser metade para mater $y$ e
     - se $|E_{ij}| >> |X_{ij}|$, tem-se que $(X+E) \approx E$  e toda a porção direita dentro dos colchetes da equação 3 se aproxima da identidade,
        de modo que $\hat{\Theta}_E \approx 0_{2 \times 1}$.
    
    Resumindo: a presença de ruídos em $X$ tende a reduzir a magnitude da soluçãi estimada. Esse resultado faz sentido
    quando entendemos que ruídos reduzem a correlação entre as variáveis e, portanto, seus respectivos coeficientes angulares.

    ##### A Descida do Gradiente
     
    A Descida do Gradiente é um método de otimização que se tornou fundamental em técnicas de aprendizado de máquina, 
    com destaque às Redes Neurais. Sua aplicação consiste em evoluir uma solução - do problema de otimização - para a redução 
    de uma função perda (também chamada de função custo) estabelecida pelo par problema-modelo, a exemplo da erro quadrático médio.                

    Em nosso caso, o modelo se restringe a uma regressão linear, os parâmetros
    são $\theta_1$ e $\theta_2$ e a função perda $L$, definida abaixo,  que deverá ser minimizada.
             
    $L(\theta_1, \theta_2; n) = (1/n)\Sigma_{i=1}^{n} [(x_1\theta_1 + x_2\theta_2) - y_i]^2$, 
             
    A intuição por detrás da Descida do Gradiente está a ideia de explorar a superfície da própria função erro para direcionar 
    alterações nos parâmetros de controle até que a minimização seja alcançada. Explicitamente, a Descida do Gradiente é um método 
    iterativo que atualiza os parâmetros de um modelo a partir da seguinte regra de atualização:
    
    $\Theta_{i+1} = \Theta_{i} - \frac{\eta}{n}\frac{dL}{d\Theta} \Biggr\rvert_{\Theta=\Theta_i}$ (equação 4)
    
    O procedimento fica claro quando vemos que $\Theta$ evolui na direção oposta ao crescimento de $L(\Theta)$ e, portanto, na
    direção em que a função perda diminui! O modelo ainda adiciona o parâmetro $\eta>0$, chamado de taxa de aprendizagem, para registringir o 
    tamanho do passo dado, evitando que o processo se torne errático nos casos em que a função perda possui gradiente elevado. Todo o processo
    é comumente chamado de aprendizagem. 

    No nosso caso, uma regressão linear, a derivada de $L$ com relação a $\theta_{j}$ da equação 4 fica:
             
    $\theta_{j_{,i+1}} = \theta_{j_{,i}} - 2 \frac{\eta}{n} \Sigma_{i=1}^{n} [(x_1\theta_{1_{,i}} + x_2\theta_{2_{,i}})-y_i]x_j  $ (equação 5)
             
    em que $n$ corresponde ao numero total de amostras. 
             
    A partir de um chute inicial para $\Theta$, frequentemente escolhido aleatoriamente, o processo iterativo explicitado é realizado
    quantas vezes se queira, ou até que algum critério de convergência seja alcançado. 

    ###### Descida do Gradiente do tipo 'Lote'                      
      
    Na etapa anterior, definimos um processo de aprendizagem que utiliza de todas as amostras - $n$ - na otimização de $\Theta$. 
    Contudo, nada nos impede de utilizarmos para tal apenas uma parte do conjunto total, ou pequenos lotes (ou mini batch), e esssa mudança é o que 
    define o procedimento denominado Descida do Gradiente em Mini Lotes ou *'Mini Batch Gradient Descent'*. Essa mudança oferece uma abordagem 
    mais [eficiente em termos computacionais](https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu) e pode levar a uma convergência mais rápida, especialmente em conjuntos de dados grandes.
             
    Nesse caso, supondo que o batch/lote contém $B$ amostras, a equação 5 se torna       
    $\theta_{j_{,i+1}} = \theta_{j_{,i}} - 2 \frac{\eta}{B} \Sigma_i [(x_1\theta_{1_{,i}} + x_2\theta_{2_{,i}})-y_i]x_j$

    Um caso particular da abordagem é quando o realizamos a etapa de aprendizagem usando apenas uma simples amostra, $B=1$. Nesse caso,
    o procedimento é chamado de Descida Estocástica do Gradiente, ou *'Stochastic Gradient Descent'*. O estocástico se refere à aleatoriedade 
    adicionada à iteração quando a média do desvio quadrático é 'realizada' sobre apenas por um valor.   

    A aplicação do método consiste em realizar os seguintes passos:
             
    1. Reorganizar aleatoriamente as $n$ amostras;   
    2. Dividir os dados em lotes contíguos de dimensão $B$;          
    3. Realizar o aprendizado, ou atualizar $\Theta$, uma vez para cada um dos conjuntos de lote;             
    4. Repetir as etapas de 1 a 3 um número de vezes desejado, chamado de número de épocas, ou até que algum critério seja satisfeito.     

    As vantagens desse procedimento são:
    
    1. Eficiência Computacional: o uso de mini lotes permite explorar a paralelização; 
    2. Regularização Estocástica Natural: a introdução de aleatoriedade na escolha de mini lotes, bem como a dimensão do lote, pode agir como uma forma 
    de regularização estocástica, ajudando a evitar sobreajuste (overfitting).
    3. Melhoria na Convergência: o ruído introduzido pelos mini lotes pode ajudar a escapar de mínimos locais e a acelerar 
    a convergência.
 
    
    ##### Técnicas de Regularização
             
    Técnicas de [regularização](https://www.quora.com/What-is-the-importance-of-regularization-techniques-in-deep-learning-models-for-website-functionality) 
    são procedimentos aplicados também em problemas de otimização e se caracterizam por reduzir
    sobreajuste e melhorar a generalização de modelos. A expressão 'regularização' diz repeito à restrição imposta pelas técnicas 
    aos parâmetros de controle dos modelos, e suas formas mais comuns são as Regularização do tipo $L_1$ e $L_2$.

    ###### Regularização $L_1$, ou Lasso:
    Adiciona à função perda um termo proporcional à soma dos valores absolutos dos coeficientes. Matematicamente, a função perda é atualizada
    para :

    $L_{L1}(\theta_1, \theta_2; B) = L(\theta_1, \theta_2; B)_{original} + \lambda_1\Sigma_{j=1}^2|\theta_j|$
             
    Essa adaptação penaliza os coeficientes não nulos, levando à redução ou até anulação de alguns coeficientes. $\lambda_1$ é o 
    parâmetro de regularização.   
 
    ###### Regularização $L_2$, ou Ridge:
    Adiciona à função perda um termo proporcional à soma dos quadrados dos coeficientes. Assim, a função perda é atualizada
    para :

    $L_{L2}(\theta_1, \theta_2; B) = L(\theta_1, \theta_2; B)_{original} + \lambda_2\Sigma_{j=1}^2\theta_j^2$
             
    A mudança penaliza os coeficientes mais altos, levando a redução, mas nunca a anulação de fato, de alguns coeficientes. Nesse caso, $\lambda_2$ é o 
    parâmetro de regularização.    
             
    Notemos que se os parâmetros $\lambda_1$ e $\lambda_2$ são altos, a tendência da solução é evoluir para as proximidades de zero 
    para redução da função perda. 
             
    Agora, vamos generalizar a função perda ao considerar o método da Descida de Gradiente por Mini Lotes e adicionar regularizações para fins 
    de exemplificação do procedimento. Para tanto, chegamos a uma função perda dada por

    $L(\theta_1, \theta_2; B) = (1/B)\Sigma_{i=1}^B [(x_1\theta_1 + x_2\theta_2) - y_i]^2 + \lambda_1\Sigma_{j=1}^2|\theta_j| + \lambda_2\Sigma_{j=1}^2 \theta_j^2$
                
    Regularizações nos trazem as seguintes vantagens:

    1. Prevenção de Overfitting ou ajuste excessivo aos dados de treinamento mesmo na presença de ruído;
    2. Melhoria da Generalização, quando as penalizacoes reduzem complexidade e permitem melhor previsão para dados novos;
    3. Reduz a sensibilidade do modelo a problemas mal postos, quando há correlação entre as variáveis independentes; 
    4. Promove o equilíbrio entre viés (dados de teste) e variância (dados de treinamento)
    
             
    Na aba 'Testando Parâmetros', no topo dessa página, iremos verificar como todos esses parâmetros afetam a nossa solução.         
    ''')

with tab3:
    st.write(r'''
    ##### O código 
                
    O código cria Np amostras aleatórias que satisfazem precisamente a equação linear $y = x_1 \theta_1 + x_2 \theta_2 $, sendo 
    os parâmetros $Np$, $\theta_1$ e $\theta_2$ definidos pelo própio usuário na barra lateral à esquerda. Dada a natureza do problema,
    o par $\Theta$ = ($\theta_1$,$\theta_2$) é também chamado de solução ideal da equação linear, caso tivéssemos em mãos apenas $y$, $x_1$ e $x_2$.
             
    Em seguida, alteramos o problema adicionando ruído - de distrubuição normal centrado em zero e de variância $\sigma^2$ - a cada 
    $x_{i,j}$ criado na etapa anterior. Mais uma vez, o parâmetro $\sigma$ pode ser alterado pelo usuário. Essa alteração é suficiente para
    que o método dos mínimos quadrados levem a uma estimativa $\Theta'$ = ($\theta_1', \theta_2'$) um pouco diferente do par inicial. Essa estimativa 
    é aqui chamada de $\Theta'$. 
             
    Embora tenhamos em mãos uma estimativa de $\Theta$ tanto para os dados ideais quanto para os ruidosos, o objetivo desse código é 
    demonstrar como o método da 'Descida do Gradiente' encontra uma solução. Para tanto, é necessário se definir uma função perda, 
    que pode ser vista abaixo: 
             
    $L(\theta_1, \theta_2) = (1/B)\Sigma_{i=1}^B [(x_1\theta_1 + x_2\theta_2) - y_i]^2 + \lambda_1\Sigma_{j=1}^2|\theta_j| + \lambda_2\Sigma_{j=1}^2 \theta_j^2$,
    
    em que $B$, $\lambda_1$ e $\lambda_2$ representam o tamanho no 'batch' e os parâmetros de regularização, todos especificado 
    pelo usuário em 'Mini Batch Size'.
    
    Por dependenter de $\theta_1$ e $\theta_2$, a função perda pode ser interpertada como um mapa, e é justamente esse mapa que se vê no 
    gráfico do topo da aba "Visualização do Método", adicionado de contornos para uma melhor interpretação. Para construção do mapa, variamos ambos
    os parâmetros de -10 a 10, e a região cujo par fornece a menor função perda é aquela de tom vermelho mais escuro. 
             
    Veja, no mapa, marcado com uma estrela verde o par/solução-ideal $\Theta$ definido inicialmente pelo usuário. O mapa ainda mostra com 
    um X cinza o par $\Theta'$ estimado pelo método dos mínimos quadrados para o conjunto de dados ruidosos.  Note como o ruído afeta a solução.  

    Os parâmetros restantes são então utilizados pelo método da Descida do Gradiente para identificação da solução ótima. 
    Definidos o 'Batch Size' e o número de épocas, o processo iterativo evoluirá a solução $\Theta''$ = ($\theta_1'',\theta_2''$), que pode ser visto 
    no mapa como uma espécie de trajetória. O usuário tem a liberdade de escolher as soluções iniciais em $\theta_1(0)$ e $\theta_2(0)$ que iniciarão 
    o processo iterativo definido pela equação 5 da aba "Uma Breve Explicação". Destaca-se em com um losango vermelho a solução obtida ao fim das iterações.

    Enquanto o par ($\theta_1'',\theta_2''$) evolui, a função perda também experimenta mudanças, obviamente. Essa alteração pode ser vista na figura 
    logo abaixo ao mapa, em que o eixo x representa a iteração corrente. Note que o número máximo de iterações é dada por $Num$-$Epocas$ $\times$ 
    int($Np$/$Tamanho$-$do$-$Lote$).
    
    Se o usuário quiser manter fixa a semente do gerador de aleatórios, basta selecionar *yes* no campo *Fixed Seed?*. Caso contrário, a cada
    alteração nos parâmetros, um novo conjunto de dados será resolvido, dificultando a interpretação da influência de cada parâmetro.  

    Veja a aba "Uma Breve Explicação" para um melhor entendimento a cerca dos parâmetros.

    ''')


