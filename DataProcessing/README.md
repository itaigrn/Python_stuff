Data proccesing and upload webpage for expirements results file and storing proccessed results in a sql file.

How to run:
1. after the zip file has been unzipped open this directory's files as a project in an IDE and install uv with 'pip install uv' in the terminal
2. enter the command line 'uv run app.py'
3. a link to the relevant URL will open up - click on it, and you will be able to upload files
4. in the uploads directory you can see examples of valid and invalid files of both formats and can try to upload them on the web page.
5. You will be able to see the formulation_ids that were not valid after upload (if file uploaded is invalid)
or if all formulations in the file were valid - no additional comments will appear
6. After successful upload of a valid file you can access the calculated values and formulation id's uploaded so far 
and also stats about each type of experiment/instrument by pressing the relevant button. 

assumptions:
* in TNS experiment type, since the name of the formulation (AKA formulation_id) is not explicitly mentioned, I took the liberty to create a counter
that will distribute a unique natural number for each formulation sequentially.

for further questions you can try my email itaigrn286@gmail.com
