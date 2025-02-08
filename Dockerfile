# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create Conda environment
RUN conda env create -f environment.yml

# Activate environment by default
SHELL ["conda", "run", "-n", "waterfall", "/bin/bash", "-c"]

# Copy the project files
COPY . .

# Set environment variables if needed
ENV PYTHONPATH=/app

# The CMD instruction can be overridden when running the container
# For development, you can use bash to explore the environment
CMD ["/bin/bash"]
