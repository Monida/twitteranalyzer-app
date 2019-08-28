# twitteranalyzer-app

-------------------------------------------------FILE UNDER CONSTRUCTION------------------------------------------------------


This repository contains all the files of the Web-app TwitterAnalyzer, which you can try out on this link: https://twitteranalyzer-app.herokuapp.com/

Below you can find the description to the content of this repository. For those of you who want to get to the point you can go the "Short description" section. However, if you want to know in detail how I created this proyect jump to the "Detail description" section.


**SHORT DESCRIPTION**

**--Project folders structure--**

twitteranalyzer-app/

    static/
    
      stlye.css
      
      keyword_list
      
    templates/
    
      enterquery.html
      
      analyzing.html
      
      returnquery.html
      
      error.html
      
    twitter_analyzer/
    
      __init__.py
      
      Twitter.py
      
    .gitignore
    
    app.py
    
    Procfile
    
    Procfile.windows
    
    README
    
    requirements
    
    runtime
    

**--Project folders description--**

**+ twitteranalyzer-app:** This is the project folder. All files needed to run the app must be stored here. 

  **+ static/:** This static folder contains all files needed by either the html files or the app itsealf to work.
  
   **+  style.css:** This file contains the CSS code that gives format to all the html templates.
      
   **+ keyword_list:** One of the functions of the TwitterAnalyzer is to find the topics that are being discussed and label them. So far it does that by comparingthe most frecuent words to the list of words (values) corresponding for each keyword (key) of the dictionary.
  
  **+ templates/:** This folder holds all the html templates. 

   **+  analyzing.html:** This file appears while the program downloads the tweets and analyze them to indicate the user that the data is being analyzed and the results will show up shortly. It is currently not implemented yet.
    
   **+  enterquery.html:** This file displays the instructions to use the twitteranalyzer-app. It also contains the form for the user to enter a query as well as a search button.
  
   **+  error.html:** This html file is showed when the app presents an error to provide the user with a corresponding message.
  
   **+  moreinsights:** This html is called when in the returnquery.html the user clic on a specific topic to know more insights about it. 
  
   **+  returnquery.html:** This is the main page that is shown after the program analyzes the tweets. It displays a summary about the tweets that were found.
  
  **+  twitter_analyzer/:** This folder holds the files of the twitter_analyzer package

   **+  Twitter.py:** This file contains the class Twitter which is used by the app to download, clean, analyze and display the tweets and the corresponding analysis. 
  
   **+  __init__:** This file indicates that the directory twitter_analyzer is a python package.
  
  **+  .gitignore:** This hidden text file contains the files and directories that are not tracked using git.
  
  **+  app.py:** This is the main script responsible for running the app. 
   
  **+  nltk.txt:** This text file is necesary to deploy to Heroku an app that uses NLTK. It contains the corpora that need to be downloaded for the app to work. During the app build up the corpora is downloaded and installed. To learn more about it go to this link: https://devcenter.heroku.com/articles/python-nltk.
  
  **+  Procfile:** This is a file *with no extension* that contains the commands that need to be execuded during the app startup. Go to the following link to learn more about it: https://devcenter.heroku.com/articles/procfile.
  
  **+  Procfile.Windows:** This is also a file with no extension and just like Procfile it tells the app what code to run during start up, but it is an extra file needed when the app is built on Windows. 
  
  **+  README.md:** This file (the current file) contains the entire description of the twitteranalyzer-app project.
    
  **+  requirements.txt:** Heroku needs this requirements file to be able to install all the libraries that are necesary to be able to run the app. 
  
  **+  runtime.txt:** This files to the app what version of python is needed.
    
  **+  worker.py:** When there are some lengthy processes in a Heroku app, it is necesary to perform these processes asynchronously from the main running time to make the use of the app more efficient. Furthermore, if these lengthy tasks take more than 30 seconds the Heroku server will return a timeout error making the app to crash. The worker file will help the app queue the lengthy tasks to be run in the background. If you want to learn how this is implemented in link: https://devcenter.heroku.com/articles/python-nltk. This part of the project is currently being implemented.   
      

**DETAIL DESCRIPTION**
