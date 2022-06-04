import glob
import argparse
import os 
parser = argparse.ArgumentParser(description='Convert all obj to gltf via assimp')
parser.add_argument('directory', type=str,
                    help='Path to the folder with obj files to be rendered.')
parser.add_argument('output', type=str,
                    help='Output file for caching the list of files')
args = parser.parse_args()

files = glob.glob(args.directory + '/**/*.obj', recursive=True)
out = open(args.output, "w")
for f in files:
    out.write(os.path.abspath(f)+"\n")