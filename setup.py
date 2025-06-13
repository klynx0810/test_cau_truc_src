from setuptools import setup, find_packages

# print(find_packages(where="."))

setup(
    name='neuroflow',
    version='0.1.0',
    author='Lê Trung Kiên',
    author_email='21011496@st.phenikaa-uni.edu.vn',
    description='Thư viện mô phỏng kiến trúc TensorFlow chỉ dùng NumPy',
    # packages=[
    #     "neuroflow",
    #     "neuroflow.layers",
    #     "neuroflow.models",
    #     "neuroflow.losses",
    #     "neuroflow.optimizers",
    #     "neuroflow.saving",
    #     "neuroflow.src",
    # ],
    packages=find_packages(where="."),
    package_dir={"neuroflow": "neuroflow"},   
    install_requires=["numpy"],
)
