from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv

load_dotenv()

model1=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=300)

model2=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=500)

prompt1=PromptTemplate(
    template='Generate paragraph on the topic {topic}',
    input_variables=['topic']
)


prompt2=PromptTemplate(
    template='Generate a quiz on {topic} with 3 questions',
    input_variables=['topic']
)


prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document /n notes->{notes} and quiz->{quiz}',
    input_variables=['notes','quiz']
)


parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = '''
Overview
Kubernetes is a portable, extensible, open source platform for managing containerized workloads and services that facilitate both declarative configuration and automation. It has a large, rapidly growing ecosystem. Kubernetes services, support, and tools are widely available.
This page is an overview of Kubernetes.

The name Kubernetes originates from Greek, meaning helmsman or pilot. K8s as an abbreviation results from counting the eight letters between the "K" and the "s". Google open sourced the Kubernetes project in 2014. Kubernetes combines over 15 years of Google's experience running production workloads at scale with best-of-breed ideas and practices from the community.

Why you need Kubernetes and what it can do
Containers are a good way to bundle and run your applications. In a production environment, you need to manage the containers that run the applications and ensure that there is no downtime. For example, if a container goes down, another container needs to start. Wouldn't it be easier if this behavior was handled by a system?

That's how Kubernetes comes to the rescue! Kubernetes provides you with a framework to run distributed systems resiliently. It takes care of scaling and failover for your application, provides deployment patterns, and more. For example: Kubernetes can easily manage a canary deployment for your system.

Kubernetes provides you with:

Service discovery and load balancing Kubernetes can expose a container using a DNS name or its own IP address. If traffic to a container is high, Kubernetes is able to load balance and distribute the network traffic so that the deployment is stable.
Storage orchestration Kubernetes allows you to automatically mount a storage system of your choice, such as local storage, public cloud providers, and more.
Automated rollouts and rollbacks You can describe the desired state for your deployed containers using Kubernetes, and it can change the actual state to the desired state at a controlled rate. For example, you can automate Kubernetes to create new containers for your deployment, remove existing containers and adopt all their resources to the new container.
Automatic bin packing You provide Kubernetes with a cluster of nodes that it can use to run containerized tasks. You tell Kubernetes how much CPU and memory (RAM) each container needs. Kubernetes can fit containers onto your nodes to make the best use of your resources.
Self-healing Kubernetes restarts containers that fail, replaces containers, kills containers that don't respond to your user-defined health check, and doesn't advertise them to clients until they are ready to serve.
Secret and configuration management Kubernetes lets you store and manage sensitive information, such as passwords, OAuth tokens, and SSH keys. You can deploy and update secrets and application configuration without rebuilding your container images, and without exposing secrets in your stack configuration.
Batch execution In addition to services, Kubernetes can manage your batch and CI workloads, replacing containers that fail, if desired.
Horizontal scaling Scale your application up and down with a simple command, with a UI, or automatically based on CPU usage.
IPv4/IPv6 dual-stack Allocation of IPv4 and IPv6 addresses to Pods and Services.
Designed for extensibility Add features to your Kubernetes cluster without changing upstream source code.


'''
result=chain.invoke({'topic':text})

print(result)

chain.get_graph().print_ascii()

'''
# Kubernetes: Overview and Assessment

## Introduction to Kubernetes
Kubernetes is a powerful, portable, and open-source platform designed to automate the deployment, scaling, and management of containerized workloads. Originally open-sourced by Google in 2014, its name stems from the Greek word for "helmsman" or "pilot"—often abbreviated as **K8s**—and builds upon over a decade and a half of Google's industry experience running massive production workloads.

In modern production environments where application downtime is unacceptable, Kubernetes serves as a resilient framework that effortlessly handles complex operational challenges.

## Key Features
Kubernetes offers a comprehensive suite of features, including:
*   **Automated Rollouts and Rollbacks:** Manages changes to your applications.
*   **Service Discovery and Load Balancing:** Exposes containers using DNS names or their own IP addresses.
*   **Automatic Bin Packing:** Optimally utilizes resources by fitting containers onto nodes based on CPU and memory needs.
*   **Self-Healing:** Automatically restarts, replaces, or kills containers that fail or do not respond to health checks.
*   **Secret and Configuration Management:** Securely stores and manages sensitive information.
*   **Horizontal Scaling:** Adjusts the number of running containers based on demand.
*   **Storage Orchestration:** Seamlessly manages storage across various cloud and local providers.
*   **Networking:** Supports IPv4/IPv6 dual-stack networking.

Built with extensibility at its core, Kubernetes allows developers to expand its functionality without altering upstream source code, making it the industry-standard foundation for running scalable and reliable distributed systems.

***

## Kubernetes Overview Quiz

**1. What is the origin and meaning behind the abbreviation "K8s"?**
A) It refers to the 8 core features of the platform.
B) It stands for "K-eight-services," representing the eight main cloud providers it supports.
C) It represents the eight letters that appear between the "K" and the "s" in the word Kubernetes.
D) It was named after the 8th iteration of Google’s internal container project.

**2. Which Kubernetes feature allows the system to automatically restart failed containers and replace those that do not respond to health checks?**
A) Service discovery
B) Self-healing
C) Storage orchestration
D) Automatic bin packing
                    +---------------------------+
                    | Parallel<notes,quiz>Input |
                    +---------------------------+
                       ***                   ***
                   ****                         ****
                 **                                 **
    +----------------+                          +----------------+
    | PromptTemplate |                          | PromptTemplate |
    +----------------+                          +----------------+
             *                                           *
             *                                           *
             *                                           *
+------------------------+                  +------------------------+
| ChatGoogleGenerativeAI |                  | ChatGoogleGenerativeAI |
+------------------------+                  +------------------------+
             *                                           *
             *                                           *
             *                                           *
    +-----------------+                         +-----------------+
    | StrOutputParser |                         | StrOutputParser |
    +-----------------+                         +-----------------+
                       ***                   ***
                          ****           ****
                              **       **
                    +----------------------------+
                    | Parallel<notes,quiz>Output |
                    +----------------------------+
                                   *
                                   *
                                   *
                          +----------------+
                          | PromptTemplate |
                          +----------------+
                                   *
                                   *
                                   *
                      +------------------------+
                      | ChatGoogleGenerativeAI |
                      +------------------------+
                                   *
                                   *
                                   *
                          +-----------------+
                          | StrOutputParser |
                          +-----------------+
                                   *
                                   *
                                   *
                      +-----------------------+
                      | StrOutputParserOutput |
                      +-----------------------+
'''