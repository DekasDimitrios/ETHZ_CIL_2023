## Project File Structure

The file structure of the project is the following:
- src: directory containing the source code necessary to execute the project
- notebooks: a folder containing the notebooks used to deploy the project to google colab
- data: a folder containing all the data needed to run the project and also the one's produced during runs
- configs: a folder containing the configs used to pass all the appropriate parameters to the program
- runs: a folder containing the output produced by the transformer experiments excecuted

## Executing Project

In order to execute the project's code one has to follow the steps below:
- Create a conda environment by using the following installation commands:
   - ```conda create --name cil python=3.9```
   - ```conda install numpy```
   - ```conda install pandas```
   - ```conda install tqdm```
   - ```conda install nltk```
   - ```conda install yacs```
   - ```conda install jsonlines```
   - ```conda install -c anaconda scikit-learn```
   - ```conda install -c anaconda gensim=4.1.2```
   - ```conda install -c conda-forge xgboost```
   - ```conda install -c conda-forge keras```
   - ```conda install -c conda-forge tensorflow```
   - ```conda install -c conda-forge transformers```
   - ```pip install accelerate```
   - ```pip install fasttext```
   - ```pip install ekphrasis -U```
   - ```pip install nlp_dedup```
  
  Although providing a unified file that would be able to recreate our environment that 
  ability was hindered by the lack of conda support for various packages as shown above
- Download and place the data provided for the project in the data/ folder
- Download the additional data from https://www.kaggle.com/datasets/kazanova/sentiment140 and place them in the data/additional/ folder
- Download the GloVe tweet embeddings (glove.twitter.27B.zip) from https://nlp.stanford.edu/projects/glove/ and place them in the data/additional/ folder
- Specify the config setups you wish to execute
- Navigate to src/\_\_init__.py and decide if a baseline or a novelty run is being performed
- Execute the main function found in src/\_\_init__.py, either from terminal or an IDE software.