//+------------------------------------------------------------------+
//|                                                       neural.mq4 |
//|                        Copyright 2016, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

#property copyright "Copyright 2016, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <Fann2MQL.mqh>

// o n�mero total de camadas, aqui, h� uma camada de entrada,
// 2 camadas ocultas e uma camada de sa�da = 4 camadas.
int nn_layer = 4;
int nn_input = 3; // N�mero de neur�nios de entrada. Nosso teste padr�o � feito de 3 n�meros, 
                  // significando 3 neur�nios de entrada.
int nn_hidden1 = 8; // n�mero de neur�nios na primeira camada oculta
int nn_hidden2 = 5; // n�mero na segunda camada oculta
int nn_output = 1; // n�mero de sa�das

// trainingData[][] conter� os exemplos 
// Vamos usar para ensinar as regras aos neur�nios.
double      trainingData[][4];  // IMPORTANTE! size = nn_input + nn_output

int maxTraining = 500;  // n�mero m�ximo de tempo do treinamento, 
                        // os neur�nios com alguns exemplos
double targetMSE = 0.002; // o MSE (Mean-Square Error /Erro Quadr�tico M�dio) dos neur�nios, dever�amos 
                          // obter no m�ximo (voc� vai entender isso mais abaixo no c�digo)

int ann; // Esta vari�vel ser� o identificador da rede neuronal.



//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart(){
//---
   int i;
   double MSE;
   
    // N�s redimensionamos o array trainingData, para que possamos us�-lo.
   // N�s vamos mudar o seu tamanho em um de cada vez.
   ArrayResize(trainingData,1);
   
   Print("##### INIT #####");
   
   // Criamos novas redes de neur�nios
   ann = f2M_create_standard(nn_layer, nn_input, nn_hidden1, nn_hidden2, nn_output);
   
   // Vamos verificar se foi criado com sucesso. 0 = OK, -1 = erro
   debug("f2M_create_standard()",ann);
   
   // N�s definimos a fun��o de ativa��o. N�o se preocupe com isso. Apenas fa�a.
        f2M_set_act_function_hidden (ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
        f2M_set_act_function_output (ann, FANN_SIGMOID_SYMMETRIC_STEPWISE);
        
        // Alguns estudos mostram que estatisticamente os melhores resultados s�o alcan�ados usando este intervalo; 
     // mas voc� pode tentar diferente e ver se fica melhor ou o pior
        f2M_randomize_weights (ann, -0.77, 0.77);
        
        // Eu s� imprimi no console o n�mero de neur�nios de entrada e sa�da. 
      // Apenas para verificar. Apenas para fins de depura��o.
   debug("f2M_get_num_input(ann)",f2M_get_num_input(ann));
   debug("f2M_get_num_output(ann)",f2M_get_num_output(ann));
        
   
   Print("##### REGISTER DATA #####");
   
   //Agora n�s preparamos alguns exemplos de dados (com expectativa de sa�da) 
   // e os adicionamos ao conjunto de treinamento.
   // Uma vez que temos de adicionar todos os exemplos que queremos, n�s vamos enviar 
   // estes dados de treinamento configurados aos neur�nios, para que eles possam aprender.
   // prepareData() tem alguns argumentos:
   // - A��o para fazer (treinar ou computar)
   // - os dados (aqui, 3 dados por configura��o)
   // - O �ltimo argumento � a expectativa de sa�da.
   // Aqui, esta fun��o leva os dados de exemplo e a expectativa de sa�da, 
   // e adiciona nas configura��es de aprendizado.
   // Verifique o coment�rio associado a esta fun��o para obter mais detalhes.
   //
   // Aqui � o padr�o que estamos ensinando:
   // Existem 3 n�meros. Vamos cham�-los de a, b e c.
   // Voc� pode raciocinar com esses n�meros como sendo coordenadas de vetor 
  // Por exemplo (vetor indo para cima ou para baixo)
   // Se a < b && b < c ent�o sa�da = 1
   // Se a < b && b > c ent�o sa�da = 0
   // Se a > b && b > c ent�o sa�da = 0
   // Se a > b && b < c ent�o sa�da = 1
   
   
   // UP UP = UP / Se a < b && b < c ent�o sa�da = 1 
   prepareData("train",1,2,3,1);
   prepareData("train",8,12,20,1);
   prepareData("train",4,6,8,1);
   prepareData("train",0,5,11,1);

   // UP DOWN = DOWN / Se a < b && b > c ent�o sa�da = 0
   prepareData("train",1,2,1,0);
   prepareData("train",8,10,7,0);
   prepareData("train",7,10,7,0);
   prepareData("train",2,3,1,0);

   // DOWN DOWN = DOWN / Se a > b && b > c ent�o sa�da = 0
   prepareData("train",8,7,6,0);
   prepareData("train",20,10,1,0);
   prepareData("train",3,2,1,0);
   prepareData("train",9,4,3,0);
   prepareData("train",7,6,5,0);

   // DOWN UP = UP / Se a > b && b < c ent�o sa�da = 1
   prepareData("train",5,4,5,1);
   prepareData("train",2,1,6,1);
   prepareData("train",20,12,18,1);
   prepareData("train",8,2,10,1);
   
   // Agora imprimiremos a forma��o integral configurada ao console, para verificar como se parece.
   // Apenas para fins de depura��o.
  
  // printDataArray();
   
   
   Print("##### TRAINING #####");
   
   // Precisamos treinar os neur�nios muitas vezes, ordenadamente, 
   // para que sejam bons naquilo que foram solicitados para fazer.
   // Aqui vou trein�-los com os mesmos dados (nossos exemplos) v�rias vezes, 
   // at� que compreendam plenamente as regras que estamos tentando ensin�-los, ou at� 
   // O treinamento ser repetido o n�mero 'maxTraining' de vezes  
   // (neste caso maxTraining = 500)
   // Quanto melhor for entendida a regra, menor ser� o Erro Quadr�tico M�dio.
   // A fun��o de teach() retorna o Erro Quadr�tico M�dio (ou MSE)
   // 0.1 ou inferior � um n�mero suficiente para regras simples
   // 0,02 ou menor � melhor para regras complexas como a que 
   // estamos tentando ensin�-los (� um reconhecimento modelo, o que n�o � t�o f�cil. )
   for (i=0;i<maxTraining;i++) {
      MSE = teach(); // Toda vez que o loop � executado, a fun��o teach � ativada. 
                     // Confira os coment�rios associados a esta fun��o para entender mais.
      if (MSE < targetMSE) { // Se o MSE � menor do que n�s definimos (aqui targetMSE = 0,02)
         debug("training finished. Trainings ",i+1); // Ent�o imprimimos o console 
                                                     // em quantos treinamentos 
                                                     // forem necess�rios para os neur�nios compreenderem
         i = maxTraining; // e vamos sair deste ciclo
      }
   }
   
   // N�s imprimimos no console o valor MSE ap�s o treinamento ter sido conclu�do.
   debug("MSE",f2M_get_MSE(ann));
   
   
   Print("##### RUNNING #####");
   // E agora podemos solicitar aos neur�nios a an�lise de novos dados que eles nunca viram.
   // Ser� que eles v�o reconhecer os padr�es corretamente?
   // Voc� pode ver que eu usei a mesma fun��o prepareData() aqui, 
   // com o primeiro argumento definido para "compute".
   // O �ltimo argumento foi dedicado � expectativa de sa�da, 
   // quando utilizamos esta fun��o para registrar exemplos anteriores
   // � in�til, ent�o deixamos a zero.
   // Se voc� preferir, voc� pode chamar diretamente a fun��o compute().
   // Neste caso, a estrutura � computar (inputVector[]);
   // Ent�o, ao inv�s de prepareData ("compute",1,3,1,0); voc� faria algo como:
   // double inputVector[]; // declara um novo array.
   // ArrayResize(inputVector,f2M_get_num_input(ann)); 
   // Redimensionar o array ao n�mero de entrada do neur�nio.
   // inputVector[0] = 1; // Adiciona os dados no array. 
   // inputVector[1] = 3;
   // inputVector[2] = 1;
   // result = compute(inputVector); // Chamar a fun��o compute(), com o array de entrada.
   // a fun��o prepareData() chama a fun��o compute(), 
   // que imprime o resultado no console, 
   // afim de verificarmos se os neur�nios estavam certos ou n�o.
   
   
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
** Imprimir os dados utilizadas no treinamento dos neur�nios
** Este � in�til. Apenas criado para fins de depura��o.
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
** Prepara os dados para um treinamento ou computa��o.
** coloca os dados num array 
** e os envia � fun��o de treinamento ou execu��o.
** Atualiza de acordo com o n�mero de entrada/sa�da que o seu c�digo precisa.
*************************/
void prepareData(string action, double a, double b, double c, double output) {
   double inputVector[];
   double outputVector[];
   // n�s redimensionamos os array no tamanho certo
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
   // Se voc� tiver mais do que 3 entradas, basta alterar a estrutura desta fun��o.
}


/*************************
** addTrainingData()
** Adiciona um �nico conjunto de dados de treinamento 
** (exemplo dados + expectativa de sa�da) para a configura��o do treinamento global
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
** Obt�m todos os dados de treinamento e us�-os para treinar os neur�nios de uma vez.
** A fim de treinar corretamente os neur�nios, voc� precisa executar
** esta fun��o muitas vezes, at� que o Erro Quadr�tico M�dio fique abaixo do limite.
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
      //f2M_train() est� mostrando apenas um exemplo de cada vez aos neur�nios.
      call = f2M_train(ann, inputVector, outputVector);
   }
   // Uma vez que temos de mostrar um exemplo, 
   // vamos verificar se eles s�o bons, verificando o Erro Quadr�tico M�dio (MSE) dos neur�nios. 
   // Se � baixo, eles aprenderam bem!
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
   
   // Envia novos dados aos neur�nio
   out = f2M_run(ann, inputVector);
   // e verifica o que eles dizem sobre isso usando f2M_get_output().
   output = f2M_get_output(ann, 0);
   debug("Computing()",MathRound(output));
   return(output);
}


/*************************
** debug()
** Dados de impress�o ao console
*************************/
void debug(string a, string b) {
   Print(a+" ==> "+b);
}