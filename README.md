# VRAG-for-Biblio
A Vision RAG pipeline featuring a web-based graphical interface built on **Qwen2-VL** and **Colpali**.  

To set up, use the `create_new_database(index_name)` function from `rag_utils`, where `index_name` is the path to the folder containing all your PDFs.  

Once the database is created, simply run `app.py` to launch the web interface.  

**Prerequisite:** Install **Poppler** using the following command: conda install -c conda-forge poppler
