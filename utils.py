import numpy as np
import matplotlib.pyplot as plt

def visualize_errors(loss_hist: list):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist, marker='v', color='b', linestyle='-', linewidth=2, label='Значение ошибки')  
    plt.xlabel('Итерации', fontsize=12)  
    plt.ylabel('Значение ошибки', fontsize=12)  
    plt.title('График функции ошибок', fontsize=16)  
    plt.legend()  
    plt.grid(True, linestyle='--', alpha=0.6)  
    plt.xticks(fontsize=10)  
    plt.yticks(fontsize=10)  
    plt.show()  