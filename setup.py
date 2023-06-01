# Copyright 2023 Malte Mechtenberg
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from setuptools import setup



setup( name="hom_iso_unbound"
        , version="0.2.1"
        , author="Malte Mechtenberg"
        , license="Apache License, Version 2.0"
        , packages=["emg_hom_iso_unbound"]
	, python_requires='==3.8.*'
	, install_requires = [
		"tensorflow==2.8.*"
		, "numpy==1.23.3"
		, "vtk==9.0.3"
		, "evtk"
		, "ray==1.12.0"
		, "pandas==1.5.3"
		, "matplotlib"
		, "numba==0.56.4"
		, "scipy==1.9.1"
		, "tqdm"
		, "traits==6.4.1"
		, "traits-stubs==6.4.0"
		, "typing"
	])
