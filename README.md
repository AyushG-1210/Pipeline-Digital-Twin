# Major Projor : Pipeline Digital Twin

## Working of Components:


## Notes:
- All research should be **Texas specific**. Our policies and regulations are different from other states, so we need to focus on Texas, since it has the most open data and is the most relevant to us.
- Before making any changes to the README, fetch the latest version ```git fetch origin master``` and then make changes. Or else we'll have merge conflicts.
- If you already have changes made to your local and you dont want to lose it, use ```git stash```, then ```git pull origin master```, then ```git stash apply```. This will add your changes to the latest version on the repo.
- You can make your own branches if you want.

## Progress updates: [Phase 1 - 12/02/2026]

### Ayush:
- [Inspection Codes for solutions](https://www.nrc.gov/docs/ML1233/ML12339A557.pdf)
- [Pipe Standards](https://pandapipe.com/wp-content/uploads/2024/05/API-5L-X65-Standards-Pipe-Chart-Steel-Pipe-SizesThinchnessWeight.pdf)
- [Pipe standards 2](https://amerpipe.com/products/api-5l-pipe-specifications/)
- [Edge cases, pipeline near electric power lines](https://ingaa.org/wp-content/uploads/2015/10/24732.pdf)
- [Material selection and H2S safety](https://farsi.msrpco.com/wp-content/uploads/2019/05/standard-nace-mr0175.pdf)
 - [H2S guidelines](https://niobium.tech/-/media/niobiumtech/attachments-biblioteca-tecnica/nt_nace_mr0175-does-it-work-for-you.pdf)
 - [Operating pressure guidelines](https://ttwiki.azurewebsites.net/wiki/pipeline-hub-user-resources/external-corrosion-direct-assessment-procedure-rstreng/additional-information/the-development-of-the-modified-b31g-criterion-rstreng/)
 - [Pipeline health](https://www.tandfonline.com/doi/epdf/10.1080/23311916.2019.1663682?needAccess=true)
 - [Pipeline inspection](https://www.ipgmservicios.com/wp-content/uploads/2024/03/API-570-4th-Ed.2016-Addendum-2-2018-Piping-Inspection-Code.pdf)


### Swaraag:
- GIS Data and Shapefiles (Spatial Data) - https://gis.rrc.texas.gov/gisviewer/
- Pipeline Permits (T-4) - https://rrcsearch3.neubus.com/esd3-rrc/index.php?_module_=esd&_action_=keysearch&profile=12
  (How to search for T4 Permits - https://www.rrc.texas.gov/pipeline-safety/permitting-and-mapping/permitting/how-to-search-for-t-4-permits/?utm_source=chatgpt.com)
- Inspection & Historical Data : There is no single dataset showing inspection history. The Public GIS viewer has data about version history (active, inactive, abandoned).
- Datasets available to download - https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/?utm_source=chatgpt.com


### Aditya:
- Raw Data Downloaded
  - Dataset 1 — Gas Incident Flagged Files
  - Dataset 2 — Hazardous Liquid Incident Flagged Files
  - Annual Pipeline System Data
- Cleaning remaining
- Not sure about the Hazardous Liquid Incident Flagged Files , need to check in the downloaded raw files.

## Progress updates: [Phase 2 - 17/02/2026]

### Ayush:
> Task : Clean raw data and extract relevant features for the model.
- There's some finanacial data there, we can add a feature to predict the financial impact of a pipeline failure, which can be useful for risk assessment and decision making.
- Keep natural causes in mind, and we can add that to the GIS data to model the incidents better. 
- Oil and natural gas are the ones we'll focus on, both models need to accomodate to different contraints.
- Need to use all data available, only TX data will not suffice for inferences.
- Might need to use purely physiccal and chemical constants since thers not enough useable data for the ML model, and then use the ML model to predict the risk of failure based on those constants and the GIS data.
>Currently working on a dashboard to visualize data and make connections faster, so that I dont have to manually go thorugh all the data and find the heading meanings. We can integrate this into the final product as well, so that users can easily understand the data and make informed decisions. Finally made some progress lmao.

### Swaraag:
> Task : Clean GIS data and shapefiles, and extract relevant features for the model.
- Website for all Pipeline layers, Oil & gas wells, Survey boundaries, Operator facilities : https://www.rrc.texas.gov/resource-center/research/data-sets-available-for-download/
- some drawbacks of these datasets are : all the data is static and there's no data on Real-time sensor data, Pipeline condition / inspection data among some others as well. Acc to the requirement on the dashboard we will need to work our way around it. Rest everything is present on the website.
- Yet to make coordinate mappings

### Aditya:
> Task : Create a RAG pipeline to extract information from the standards and guidelines documents.


## Progress updates: [Phase 3 - 15/03/2026]
### Ayush:
> Task : Integrate the cleaned data and the RAG pipeline to create a PINN model.

### Aditya:
>Task : Upgrade RAG pipeline to include web scraping for latest/unknown questions. 
- Dont forget to mention the working of the RAG pipeline in the README in the "Working of Components" section.
