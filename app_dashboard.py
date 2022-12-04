import streamlit as st
import pickle
import pandas as pd
import numpy as np

import time
import shap


def main():
    
    # Cette instruction va permettre de s'assurer que les graphiques SHAP apparaitront correctement sur le dashboard
    shap.initjs()

    st.title("Prêt à dépenser : SCORING CREDIT")
    st.subheader("Auteur: Meri-Nut ZAGO")
    
    #Fonction d'importation des données
    @st.cache(persist=True) #permet de ne pas utiliser trop de mémoire
    def load_data():
        data = pd.read_csv('C:/Users/Lenovo/Desktop/P7_Zago_Meri-nut/X_test_sample.csv')
        return data
    
      
    def display_customer_selectbox():
    # Cette fonction permet d'afficher la boîte de sélection des clients après avoir récupéré la liste de leurs identifiants.
    
        st.sidebar.text_input("Saisir l'identifiant d'un client", value=st.session_state['customer_id'],
                          key='customer_id')
        
if __name__ == '__main__':
    main()