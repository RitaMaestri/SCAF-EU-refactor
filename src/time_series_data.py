import numpy as np
import pandas as pd
from data_closures import N, calibrationDict
from deepdiff import DeepDiff

def growth_ratio_to_rate(ratio): #the first element of ratio is 1 (at calibration year)
    growth_rates=np.array([np.nan]*(len(ratio)-1))
    growth_rates[0] = ratio[1]-1
    for t in range(1,len(ratio)-1):
        growth_rates[t]= ratio[t+1] /np.prod(1+growth_rates[:t])-1
    return growth_rates
#to be done: define growth_rate_to_ratio

# BUILD DATAFRAME

class sys_df:

    #it creates a dicitonary of parameters' temporal series based on growth rates wrt year 0 or growth ratios
    def evolve_par(self):
        
        growth_dictionary = {key: None for key in self.growth_ratios.keys()}

        step=self.years[1]-self.years[0]


        for i in self.growth_ratios.keys():
            #if i=="aYij":
            #    a=1
            par_t0=self.calib_par_dict[i] 
            gr_j=self.growth_ratios[i]

            #sto evolvendo una variabile scalare
            if not isinstance(par_t0, np.ndarray):
                growth_dictionary[i]=gr_j*par_t0
            else: 
                #growth dictionary has the same dimensions as gr_j
                par_t0=par_t0[~np.isnan(par_t0)]
                if gr_j.ndim == 1:
                    growth_dictionary[i]=gr_j*par_t0
                else:
                    repeated_array = np.tile(par_t0, (np.shape(gr_j)[1], 1)).T
                    growth_dictionary[i] = gr_j*repeated_array

        return growth_dictionary

    ####  X axis  ####
    
    # define closure : "johansen" , "neoclassic", "kaldorian", "keynes-marshall", "keynes", "keynes-kaldor","neokeynesian1", "neokeynesian2"   ########

    # fill in dataframe at time t
    def empty_dataframe(self,dictionary):
        # build X index
        length_dict = {key: np.prod(np.shape(value)) for key, value in dictionary.items()}
        
        Xindex=list([])
        [ Xindex.extend([key]*int(value)) for key, value in length_dict.items()]
        
        #build empty dataframe
        df = pd.DataFrame(data=None, index=Xindex, columns=self.years)
        
        return df

    def __parameters_dynamics(self):
        static_par = list(set(self.calib_par_dict.keys()) - set(self.dynamic_parameters.keys()))
        dynamic_par = list(self.calib_par_dict.keys() & self.dynamic_parameters.keys())
        #for 
        for key, value in {k: self.calib_par_dict[k] for k in static_par}.items():

                self.parameters_df.loc[key] = np.repeat([value.flatten() if isinstance(value, np.ndarray) else value], int(len(self.years)), axis=0).T

        for key in dynamic_par:
            if isinstance(self.calib_par_dict[key], np.ndarray):
                exo_index = ~np.isnan(self.calib_par_dict[key]).flatten()#where there are not nans it means it is exogenous. Non nans are
                
                new_df_slice=self.parameters_df.loc[key]
                
                new_df_slice[exo_index] = self.dynamic_parameters[key]
                
                self.parameters_df.loc[key] = new_df_slice
            else:
                self.parameters_df.loc[key]=self.dynamic_parameters[key]
            


    def __initialize_variables_df(self):

        self.variables_df= self.empty_dataframe(self.calib_var_dict)
        
        #self.dict_to_df(self.calib_var_dict, self.years[0])

    def __initialize_parameters_df(self):
        
        self.parameters_df=self.empty_dataframe(self.calib_par_dict)
        
        self.__parameters_dynamics()

    def dict_to_df(self,dictionary, t):
        for key, value in dictionary.items():
            value_t=value.flatten() if isinstance(value, np.ndarray) else value
            self.variables_df[t].loc[key] = value_t
            t_idx=np.where(self.years==t)[0][0]
            self.fill_parameters_df(key,value_t,t_idx)

    def fill_parameters_df(self, key, value,t):
        index_positions = np.array([j for j, label in enumerate(self.parameters_df.index) if label == key])
        if key=="pXj":
            None
        if len(index_positions)>1:
            par_mask=self.parameters_df.loc[key].isna().any(axis=1)
            self.parameters_df.iloc[index_positions[par_mask],t]=value
        else:
            self.parameters_df.iloc[index_positions,t]=value
    
    
    def df_to_dict(self, var, t):
        calib_dict = self.calib_var_dict if var else self.calib_par_dict
        df = self.variables_df if var else self.parameters_df

        shapes = {key:np.shape(value) for key, value in calib_dict.items()}
        new_dict = calib_dict.copy()

        for key in new_dict.keys():
           
        # Extract the DataFrame variable OR parameter for the current key at time 't'
            var_t = df[t].loc[key]

            # Check if the entry is iterable (array-like)
            if hasattr(var_t, "__len__"):
                # If it's iterable, convert it to a numpy array and reshape to the stored shape
                new_dict[key] = np.array(var_t.astype(float)).reshape(shapes[key])
            else:
                # If it's not iterable, use the entry as is (scalar value)
                new_dict[key] = var_t

        return new_dict

    def evolve_K(self,t):
        self.parameters_df[t+1].loc['K'] = self.variables_df[t].loc['Knext']

    def evolve_tp(self,t):
        self.parameters_df.loc['pCtp',self.years[t+1]] = np.array(self.parameters_df.loc['pCj',self.years[t-1]])
        self.parameters_df.loc['Ctp',self.years[t+1]] = np.array(self.parameters_df.loc['Cj',self.years[t]])
        self.parameters_df.loc['Gtp',self.years[t+1]] = np.array(self.parameters_df.loc['Gj',self.years[t]])
        self.parameters_df.loc['Itp',self.years[t+1]] = np.array(self.parameters_df.loc['Ij',self.years[t]])
        self.parameters_df.loc['pXtp',self.years[t+1]] = np.array(self.parameters_df.loc['pXj',self.years[t]])
        self.parameters_df.loc['Xtp',self.years[t+1]] = np.array(self.parameters_df.loc['Xj',self.years[t]])
        self.parameters_df.loc['Mtp',self.years[t+1]] = np.array(self.parameters_df.loc['Mj',self.years[t]])
    
    def __init__(self, Years, Growth_ratios, Variables_dict, Parameters_dict, Dynamic_parameters={}):

        self.years=Years
        self.growth_ratios = Growth_ratios
        self.calib_var_dict = Variables_dict
        self.calib_par_dict = Parameters_dict
        self.dynamic_parameters = Dynamic_parameters

        #endoKnext= True if "K" not in {**self.dynamic_parameters,**self.growth_ratios}.keys() else False
        #if endoKnext:
        #    K0=np.array([np.nan]*(len(self.years)))
        #    K0[0]=self.calib_par_dict['K']
        #    self.dynamic_parameters['K']=K0
        
        #GDPreal=np.array(pd.read_csv("data/GDPreal_evolution.csv")[self.years.astype(str)].iloc[0])

        #Lgrowth=np.array( pd.read_csv("data/L_growth_ratio.csv")[self.years.astype(str)].iloc[0] )
        #L=Lgrowth*[self.calib_par_dict['L']]
        
        self.dynamic_parameters={
        
            **self.evolve_par(),
            **self.dynamic_parameters
        }
        
        self.__initialize_variables_df()
        self.__initialize_parameters_df()
        #if endoKnext:
        #    self.evolve_K(self.years[0])    
        
#system = sys_df(years,growth_rates,variables,parameters)

#DeepDiff(variables,system.df_to_dict(var=True,t=2015))

