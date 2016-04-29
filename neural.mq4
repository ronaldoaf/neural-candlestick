//+------------------------------------------------------------------+
//|                                                       neural.mq4 |
//|                                         Ronaldo Araújo de Farias |
//|                                                  http://apost.me |
//+------------------------------------------------------------------+



#property copyright "Copyright 2016, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Fann2MQL.mqh>

// o número total de camadas, aqui, há uma camada de entrada,
// 2 camadas ocultas e uma camada de saída = 4 camadas.
int nn_layer = 4;
int nn_input = 3; // Número de neurônios de entrada. Nosso teste padrão é feito de 3 números, 
                  // significando 3 neurônios de entrada.
int nn_hidden1 = 8; // número de neurônios na primeira camada oculta
int nn_hidden2 = 5; // número na segunda camada oculta
int nn_output = 1; // número de saídas

// trainingData[][] conterá os exemplos 
// Vamos usar para ensinar as regras aos neurônios.
double      trainingData[][4];  // IMPORTANTE! size = nn_input + nn_output

int maxTraining = 500;  // número máximo de tempo do treinamento, 
                        // os neurônios com alguns exemplos
double targetMSE = 0.002; // o MSE (Mean-Square Error /Erro Quadrático Médio) dos neurônios, deveríamos 
                          // obter no máximo (você vai entender isso mais abaixo no código)

int ann; // Esta variável será o identificador da rede neuronal.



//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart(){
//---
   int i;
   double MSE;
   
    // Nós redimensionamos o array trainingData, para que possamos usá-lo.
   // Nós vamos mudar o seu tamanho em um de cada vez.
   ArrayResize(trainingData,1);
   
   Print("##### INIT #####");
   
   // Criamos novas redes de neurônios
   ann = f2M_create_standard(nn_layer, nn_input, nn_hidden1, nn_hidden2, nn_output);
   
   // Vamos verificar se foi criado com sucesso. 0 = OK, -1 = erro
   debug("f2M_create_standard()",ann);
   
   // Nós definimos a função de ativação. Não se preocupe com isso. Apenas faça.
        f2M_set_act_function_hidden (ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
        f2M_set_act_function_output (ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
        
        // Alguns estudos mostram que estatisticamente os melhores resultados são alcançados usando este intervalo; 
     // mas você pode tentar diferente e ver se fica melhor ou o pior
        f2M_randomize_weights (ann, -0.77, 0.77);
        
        // Eu só imprimi no console o número de neurônios de entrada e saída. 
      // Apenas para verificar. Apenas para fins de depuração.
   debug("f2M_get_num_input(ann)",f2M_get_num_input(ann));
   debug("f2M_get_num_output(ann)",f2M_get_num_output(ann));
        
   
   Print("##### REGISTER DATA #####");
   
   //Agora nós preparamos alguns exemplos de dados (com expectativa de saída) 
   // e os adicionamos ao conjunto de treinamento.
   // Uma vez que temos de adicionar todos os exemplos que queremos, nós vamos enviar 
   // estes dados de treinamento configurados aos neurônios, para que eles possam aprender.
   // prepareData() tem alguns argumentos:
   // - Ação para fazer (treinar ou computar)
   // - os dados (aqui, 3 dados por configuração)
   // - O último argumento é a expectativa de saída.
   // Aqui, esta função leva os dados de exemplo e a expectativa de saída, 
   // e adiciona nas configurações de aprendizado.
   // Verifique o comentário associado a esta função para obter mais detalhes.
   //
   // Aqui é o padrão que estamos ensinando:
   // Existem 3 números. Vamos chamá-los de a, b e c.
   // Você pode raciocinar com esses números como sendo coordenadas de vetor 
  // Por exemplo (vetor indo para cima ou para baixo)
   // Se a < b && b < c então saída = 1
   // Se a < b && b > c então saída = 0
   // Se a > b && b > c então saída = 0
   // Se a > b && b < c então saída = 1
   
   
   // UP UP = UP / Se a < b && b < c então saída = 1 
   prepareData("train",1,2,3,1);
   prepareData("train",8,12,20,1);
   prepareData("train",4,6,8,1);
   prepareData("train",0,5,11,1);

   // UP DOWN = DOWN / Se a < b && b > c então saída = 0
   prepareData("train",1,2,1,0);
   prepareData("train",8,10,7,0);
   prepareData("train",7,10,7,0);
   prepareData("train",2,3,1,0);

   // DOWN DOWN = DOWN / Se a > b && b > c então saída = 0
   prepareData("train",8,7,6,0);
   prepareData("train",20,10,1,0);
   prepareData("train",3,2,1,0);
   prepareData("train",9,4,3,0);
   prepareData("train",7,6,5,0);

   // DOWN UP = UP / Se a > b && b < c então saída = 1
   prepareData("train",5,4,5,1);
   prepareData("train",2,1,6,1);
   prepareData("train",20,12,18,1);
   prepareData("train",8,2,10,1);
   
   // Agora imprimiremos a formação integral configurada ao console, para verificar como se parece.
   // Apenas para fins de depuração.
  
  // printDataArray();
   
   
   Print("##### TRAINING #####");
   
   // Precisamos treinar os neurônios muitas vezes, ordenadamente, 
   // para que sejam bons naquilo que foram solicitados para fazer.
   // Aqui vou treiná-los com os mesmos dados (nossos exemplos) várias vezes, 
   // até que compreendam plenamente as regras que estamos tentando ensiná-los, ou até 
   // O treinamento ser repetido o número 'maxTraining' de vezes  
   // (neste caso maxTraining = 500)
   // Quanto melhor for entendida a regra, menor será o Erro Quadrático Médio.
   // A função de teach() retorna o Erro Quadrático Médio (ou MSE)
   // 0.1 ou inferior é um número suficiente para regras simples
   // 0,02 ou menor é melhor para regras complexas como a que 
   // estamos tentando ensiná-los (é um reconhecimento modelo, o que não é tão fácil. )
   for (i=0;i<maxTraining;i++) {
      MSE = teach(); // Toda vez que o loop é executado, a função teach é ativada. 
                     // Confira os comentários associados a esta função para entender mais.
      if (MSE < targetMSE) { // Se o MSE é menor do que nós definimos (aqui targetMSE = 0,02)
         debug("training finished. Trainings ",i+1); // Então imprimimos o console 
                                                     // em quantos treinamentos 
                                                     // forem necessários para os neurônios compreenderem
         i = maxTraining; // e vamos sair deste ciclo
      }
   }
   
   // Nós imprimimos no console o valor MSE após o treinamento ter sido concluído.
   debug("MSE",f2M_get_MSE(ann));
   
   
   Print("##### RUNNING #####");
   // E agora podemos solicitar aos neurônios a análise de novos dados que eles nunca viram.
   // Será que eles vão reconhecer os padrões corretamente?
   // Você pode ver que eu usei a mesma função prepareData() aqui, 
   // com o primeiro argumento definido para "compute".
   // O último argumento foi dedicado à expectativa de saída, 
   // quando utilizamos esta função para registrar exemplos anteriores
   // é inútil, então deixamos a zero.
   // Se você preferir, você pode chamar diretamente a função compute().
   // Neste caso, a estrutura é computar (inputVector[]);
   // Então, ao invés de prepareData ("compute",1,3,1,0); você faria algo como:
   // double inputVector[]; // declara um novo array.
   // ArrayResize(inputVector,f2M_get_num_input(ann)); 
   // Redimensionar o array ao número de entrada do neurônio.
   // inputVector[0] = 1; // Adiciona os dados no array. 
   // inputVector[1] = 3;
   // inputVector[2] = 1;
   // result = compute(inputVector); // Chamar a função compute(), com o array de entrada.
   // a função prepareData() chama a função compute(), 
   // que imprime o resultado no console, 
   // afim de verificarmos se os neurônios estavam certos ou não.
   
   
   /*
   debug("1,3,1 = UP DOWN = DOWN. Should output 0.","");
   prepareData("compute",1,3,1,0);
   
   debug("1,2,3 = UP UP = UP. Should output 1.","");
   prepareData("compute",1,2,3,0);
   
   debug("3,2,1 = DOWN DOWN = DOWN. Should output 0.","");
   prepareData("compute",3,2,1,0);
   
   debug("45,2,89 = DOWN UP = UP. Should output 1.","");
   prepareData("compute",45,2,89,0);
   
   debug("1,3,23 = UP UP = UP. Should output 1.","");
   prepareData("compute",1,3,23,0);
   
   debug("7,5,6 = DOWN UP = UP. Should output 1.","");
   prepareData("compute",7,5,6,0);
   
   */
   
   debug("2,8,9 = UP UP = UP. Should output 1.","");
   prepareData("compute",10,40,50,0);
   
   Print("=================================== END EXECUTION ================================");





}
//+------------------------------------------------------------------+


/*************************
** printDataArray()
** Imprimir os dados utilizadas no treinamento dos neurônios
** Este é inútil. Apenas criado para fins de depuração.
*************************/
void printDataArray() {
   int i,j;
   int bufferSize = ArraySize(trainingData)/(f2M_get_num_input(ann)+f2M_get_num_output(ann))-1;
   string lineBuffer = "";
   for (i=0;i<bufferSize;i++) {
      for (j=0;j<(f2M_get_num_input(ann)+f2M_get_num_output(ann));j++) {
         lineBuffer = StringConcatenate(lineBuffer, trainingData[i][j], ",");
      }
      debug("DataArray["+i+"]", lineBuffer);
      lineBuffer = "";
   }
}


/*************************
** prepareData()
** Prepara os dados para um treinamento ou computação.
** coloca os dados num array 
** e os envia à função de treinamento ou execução.
** Atualiza de acordo com o número de entrada/saída que o seu código precisa.
*************************/
void prepareData(string action, double a, double b, double c, double output) {
   double inputVector[];
   double outputVector[];
   // nós redimensionamos os array no tamanho certo
   ArrayResize(inputVector,f2M_get_num_input(ann));
   ArrayResize(outputVector,f2M_get_num_output(ann));
   
   inputVector[0] = a;
   inputVector[1] = b;
   inputVector[2] = c;
   outputVector[0] = output;
   if (action == "train") {
      addTrainingData(inputVector,outputVector);
   }
   if (action == "compute") {
      compute(inputVector);
   }
   // Se você tiver mais do que 3 entradas, basta alterar a estrutura desta função.
}


/*************************
** addTrainingData()
** Adiciona um único conjunto de dados de treinamento 
** (exemplo dados + expectativa de saída) para a configuração do treinamento global
*************************/
void addTrainingData(double &inputArray[], double &outputArray[]) {
   int j;
   int bufferSize = ArraySize(trainingData)/(f2M_get_num_input(ann)+f2M_get_num_output(ann))-1;
   
   // registra os dados de entrada ao array principal 
   for (j=0;j<f2M_get_num_input(ann);j++) {
      trainingData[bufferSize][j] = inputArray[j];
   }
   for (j=0;j<f2M_get_num_output(ann);j++) {
      trainingData[bufferSize][f2M_get_num_input(ann)+j] = outputArray[j];
   }
   
   ArrayResize(trainingData,bufferSize+2);
}


/*************************
** teach()
** Obtém todos os dados de treinamento e usá-os para treinar os neurônios de uma vez.
** A fim de treinar corretamente os neurônios, você precisa executar
** esta função muitas vezes, até que o Erro Quadrático Médio fique abaixo do limite.
*************************/
double teach() {
   int i,j;
   double MSE;
   double inputVector[];
   double outputVector[];
   ArrayResize(inputVector,f2M_get_num_input(ann));
   ArrayResize(outputVector,f2M_get_num_output(ann));
   int call;
   int bufferSize = ArraySize(trainingData)/(f2M_get_num_input(ann)+f2M_get_num_output(ann))-1;
   for (i=0;i<bufferSize;i++) {
      for (j=0;j<f2M_get_num_input(ann);j++) {
         inputVector[j] = trainingData[i][j];
      }
      outputVector[0] = trainingData[i][3];
      //f2M_train() está mostrando apenas um exemplo de cada vez aos neurônios.
      call = f2M_train(ann, inputVector, outputVector);
   }
   // Uma vez que temos de mostrar um exemplo, 
   // vamos verificar se eles são bons, verificando o Erro Quadrático Médio (MSE) dos neurônios. 
   // Se é baixo, eles aprenderam bem!
   MSE = f2M_get_MSE(ann);
   return(MSE);
}


/*************************
** compute()
** Computa um conjunto de dados e retorna o resultado computado
*************************/
double compute(double &inputVector[]) {
   int j;
   int out;
   double output;
   ArrayResize(inputVector,f2M_get_num_input(ann));
   
   // Envia novos dados aos neurônio
   out = f2M_run(ann, inputVector);
   // e verifica o que eles dizem sobre isso usando f2M_get_output().
   output = f2M_get_output(ann, 0);
   debug("Computing()",MathRound(output));
   return(output);
}


/*************************
** debug()
** Dados de impressão ao console
*************************/
void debug(string a, string b) {
   Print(a+" ==> "+b);
}
