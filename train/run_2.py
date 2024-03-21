#!/usr/bin/env python
from train_2 import Train 
from utils.arg import get_args 


def main():
    args = get_args()
    trainer =  Train(args)
    trainer.fit()
    
if __name__ == '__main__':
    main() 