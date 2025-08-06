# Plant Examiner

This project provides a means to examine plants using a Gemma 3N fine tuned model. It's intended to run on a jetson agx orin. Keep in mind there are requirements specific to the version of jetpack that may cause issues without some user intervention.

Additionally the VLLM version had to be custom created as branches for pytorch 2.8.0 and the multi model support for gemma 3n were not merged at the time of creation.

You'll need to build the wheel for vllm to fully support multi model via vllm on the Jetson AGX Orin device. To do this follow these steps:

```
mkdir -p ~/src && cd ~/src
git clone --recursive https://github.com/huydhn/vllm.git vllm
cd vllm
git checkout pytorch-2.8.0

git remote add gemma https://github.com/NickLucche/vllm.git
git fetch gemma gemma3n-mm-nick

git checkout -b pytorch-2.8.0-gemma3n
git merge --no-ff gemma/gemma3n-mm-nick

# Fix the few conflicts, picking the mm branch changes
git add .

pip wheel --no-deps -w ~/wheelhouse ./src/vllm
```

To install:
`pip install -r requirements.txt`

To run the API:
`./main.sh`
To run the web server for traffic:
`python3 -m http.server 8080`
