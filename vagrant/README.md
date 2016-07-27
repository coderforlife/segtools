This directory contains a vagrant configuration file and bootstrap
to create a centos 7 virtual machine with all required packages
needed to install segtools. 

Before using install the following software in order listed:

* https://www.virtualbox.org/wiki/Downloads
* https://www.vagrantup.com/

Once the above is installed simply cd into this directory and
type **vagrant up** as shown in full example with git clone of
this repo:

```Bash
git clone https://github.com/slash-segmentation/segtools.git
cd segtools/vagrant
vagrant up
```

Once the machine is up you can ssh to it with this command from **vagrant** directory:

```Bash
vagrant ssh
```

