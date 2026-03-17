# Hybridization IOT - REMIND

## **Overview**

This project perform the hybridization of NGFS-REMIND-MAgPIE 3.2-4.6 scenario data about regional energy consumption by energy use with the energy consumption data of Input–Output Tables in the [GTAP*Agg*](https://www.centre-cired.fr/wp-content/uploads/2021/07/cired_wp_2012_39_hamdicherif_ghersi.pdf) format at the calibration year. The program produces a dataset of energy volumes and prices disaggregated both by REMIND's energy use and IOT's energy consumers through an optimization algorithm. It then projects the calibration dataset according to the REMIND trajectories and saves the resulting energy volumes and specific margin rates by consumer in a csv file of the REMIND format.

## **Pipeline Summary**

0. Preprocess IEA data
    - `run.py` executes `IEA/aggregate_region_res_agri_fish.py` to generate the IEA aggregate dataset if missing (or if forced in config).
    - the output path and filename are both declared in `config.json`.
1. Load configuration and input datasets
    - expected inputs below
1. Filter and augment REMIND data
    - I filter the REMIND dataset to retain only domestic energy-consumption variables and save the resulting subset. This reduces memory usage and significantly speeds up data loading and processing, as the full dataset is very large.
    - I augment the dataset with supplementary data from PIK.
1. Import and map IOT tables
1. Compute net energy consumption
    - I identify the energy sales tax by consumer in the IOT throug the energy_sales_tax.csv mapping. I subtract it from the gross value.
1. Rescale REMIND prices
    - REMIND prices need to be rescaled (all by the same coefficient) in order for the total energy consumption (in value) deriving from REMIND volumes and prices to match the total energy consumption of the IOT. This is hypothesis undelies the energy allocation algorithm described below.
1. Build prioritization rules
    - the energy allocation priority rules must be decided by the modeller based on avaiable supplementary data and plausibility considerations. 
    See [Building Priority Rules](#building-priority-rules)
1. Build disaggregation key: the value added is chosen. 
1. Allocate each REMIND energy use to IOT consumers in value.
    - see [Energy Allocation Algorithm](#energy-allocation-algorithm).
1. Compute volumes disaggregated by energy consumer and use from disaggregated values and REMIND energy prices.
1. Create a database adapted to host energy volumes and prices time series disaggregated by use and consumer.
1. Project energy volumes and price variables over time according to REMIND data.
1. Aggregate results by consumer and compute specific margin rates.
1. Generate final hybridized dataset as csv file.

---


## Energy Allocation Algorithm

The core algorithm disaggregates REMIND energy values into a matrix **consumer × use**. The disaggregation methodology proposed makes sure that the total consumption by consumer (in value) is consistent with the one in the IOT and the total consumption by energy use (in value) is consistent with the price-adjusted REMIND dataset. It follows **two sequential phases**:

1. **Priority-based forced allocations** (first)
2. **Constrained minimization to fill remaining cells** (second)

This ordering is important: priorities *produce fixed values* first, and the optimization only acts on the remaining free cells.



---

### **1. Priority-based forced allocations**

* A priority table (from `priorities.csv`) specifies, for some `(consumer, use)` pairs, that a fixed quantity of energy must be assigned to that consumer for that use.
* The code iterates energy-use by energy-use following a specified filling order. For each energy use:

  * allocate to the highest-priority consumer as much as allowed (consumer capacity and per-pair availability);
  * if that consumer cannot absorb the whole energy use, move the remainder to the next-priority consumer;
  * repeat until the energy type is exhausted or no further priority consumers exist.
* The result of this phase is a **matrix of forced energy values** (absolute units) and a corresponding boolean mask of fixed cells. These forced values are treated as immutable in the subsequent optimization.

> In short: priorities **fix some absolute cells** of the disaggregation matrix before any minimization happens.

---

### **2. Constrained minimization to fill remaining cells**

Only the **unfixed** cells are decision variables in the optimizer. The optimization finds values for those free cells by **minimizing the distance to a disaggregation key**, subject to hard equality constraints and bounds.

**Objective**

* Minimize the (usually quadratic) distance between the final percentages and a reference **key** (a vector of length = number of consumers).
* The key can be the same for all energy-use columns (a Series) or column-specific (a DataFrame). The optimizer uses the key *after* adjusting for forced cells (so the key is renormalized per column where forced cells exist).

**Constraints**
* TO guarantee convergence (that did not occur when using multiple disaggregated constraints), the overall minimisation constraint is the sum these vectoral constraints:
    * **Column sums:** for every energy use (column), the total allocated to consumers (sum over all the energy consumers) equals REMIND total for that use.
    * **Sector energy balances:** for every consumer (row), the sum over energy uses must equal the IOT consumption for that consumer. 
* **Fixed cells:** cells fixed by the priority phase are held constant (not variables).
* **Bounds:** free variables are bounded in ([0,1]) since they are expressed as percentages of total energy use.


### **3. Post-processing**

* An adjustment is applied to correct small residual row-sum errors (typically 10e-5) by proportionally redistributing the row error across that row's nonzero cells. This guarantees that the energy consumers totals match perfectly the IOT values and that the error is reported on REMIND volumes.

---

## Building priority rules

The modeller must chose the allocation priority rules to apply to the algorithm based on plausibility considerations and availability of external data. 

The priority dataframe hase three columns:  energy_use and energy_consumer identify the priority allocation candidate. 
The third column is the energy that is available to be allocated for the considered consumer.
For example the row BUILDINGS | HOUSEHOLDS | 0.7 indicates that the algorithm will try to allocate a 0.7 fraction of the total energy used for buildings to the households. If this fraction is larger than the households energy capacity, i.e. the consumption in the IOT, than the totality of the fraction is allocated to the households and the algorithm passes to the following declared priority. If it is smaller, the algorithm fills the whole household energy consumption capacity with energy for buildings and imposes that households will not be able to consume any other energy use.

The algorithm tries to allocate the energy to the candidates in order of appearance in the dataframe. 

If the modeller wants to compute the energy availability for some consumers and uses in the program through IEA data, they must put "to fill" in the row and modify the IEA mapping accordingly.


## **Expected input**
## Mappings
### mappings/energy_consumers.csv 
file containing 3 columns mainly used for the declaration of energy consumer categories names.
- energy consumers: contains all the energy consumers of the IOT aggregated in useful categories (typically, list of sectors, households, energy stock changes). The names given to the categories are the ones that appear in the output file.
- allocated_with_optimization_algo: boolean column that is true if the energy consumption of this consumer must be allocated with the optimization algorithm.
- allocation_exception: boolean column that is true for consumers whose energy consumption composition is handled separately (typically the stock changes).

### mappings/energy_uses.csv
file containing 2 columns mainly used for the declaration of energy uses categories names.
- energy uses: contains the energy uses present in the REMIND files aggregated to useful categories.
allocated_with_optimization_algo: boolean column that is true if the energy use must be allocated through the optimization algorithm.

### mappings/IOT-energy_consumers.xlsx
excel file with 2 windows col_label and row_label. It maps the IOT file labels to some useful variables for the programs.
 - col_labels: it maps some column labels of the IOT to the corresponding energy consumer category declared in mappings/energy_consumers.csv. The energy consumption value of each category is identified as the sum of the corresponding columns at the row identified as "ENERGY" in the window row_label.
 - row_label: it maps some row labels of the IOT to useful categories for the program. 
    - "ENERGY": the energy consumption row
    - "VA" the value added row needed to identify a disaggregation key for the optmization algorithm

 ### mappings/NGFS_energy_uses.xlsx
excel file of 3 columns used for 2 main purposes:
- mapping the REMIND dataset variable name the the respective energy use category in the program.
- mapping each energy energy use volume to the respective production price in the REMIND dataset.

### mappings/energy_sales_tax.csv
excel file that identifies the column and row labels of the energy sales taxes block in the IOT. It is needed to compute energy consumption net of taxes.

TO DO: unify the IOT block extraction in one mapping that resembles energy_sales_tax + IOT-energy_consumers

### mappings/mapping_IEA.csv
mapping between the column labels of the IEA dataset and the correspondent energy consumer. 

## Datasets
- REMIND data
- Supplementary REMIND data
- IEA data for the attribution of energy availabilities per consumer
- IOTs in the GTAP*Agg* format






## **How to Run**

```bash
python run.py
```

`run.py` first checks and prepares the IEA aggregate CSV under `IEA/result/` using the preprocessing script in `IEA/`.

## **Output**

Final dataset is stored in "out_path" contained in the config.json file.


## **Notes**

* All intermediate files are stored under `cache/`.

