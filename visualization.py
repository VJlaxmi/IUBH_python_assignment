'''
list of imports
'''
import matplotlib.pyplot as plt

def custom_scatter_plot(x,y,label_name1,**plt_kwargs):
        plt.figure(figsize=(10,5), dpi=80)
        plt.grid(True,linestyle='--')
        plt.plot(x,y)
        plt.title(label_name1)
        plt.xlabel("x")
        plt.show()
        
        
def custom_plot(x,y,x_points,y_points, label_name,n,**plt_kwargs):
        plt.figure(figsize=(10,5), dpi=80)
        plt.grid(True,linestyle='--')
        plt.scatter(x,y)
        plt.scatter(x_points, y_points,color='red',s=50)
        plt.legend( [label_name ,"Mapped Points"], ncol = 2 , loc = "lower right",bbox_to_anchor = (1.05, 1.0),frameon=False,borderaxespad=0)
        plt.title("Total Mapped points :"+n)
        plt.xlabel("x")
        plt.ylabel("Ideal Function for "+label_name)
        plt.show()