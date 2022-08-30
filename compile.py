#!/usr/bin/env python3

from idiom.cc.onnx import compile

import os
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compile the ONNX model for Envise')
    parser.add_argument('--onnx_model_path', type=str, required=True,
                        help='Path to ResNet18 model ONNX file')
    parser.add_argument('--batch-size', type=int, required=True,
                        default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()

    return args


def compile_model(onnx_model_path, batch_size):
    base_name, _ = os.path.splitext(os.path.basename(onnx_model_path))
    compiled_model_path = os.path.join(os.getcwd(), f'{base_name}-envise')

    print(f'Comping model to {compiled_model_path}')

    compile(compiled_model_path, onnx_model_path, batch_size)

    print("Compilation done")


def main():
    args = parse_arguments()
    compile_model(args.onnx_model_path, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
