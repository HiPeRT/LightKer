
<img src="img/lk-cuda-logo.png" style="height:100px">

# What it is
LightKer is a free implementation of the persistent thread paradigm for  General Purpose GPU systems. It acts as a flexible layer to "open the box" of a GPU and enables 
direct allocation of work to specific cluster(s) of GPU cores.

#What is the status?
Currently, we released the v0.2 of LightKer. It is perfectly working, please see the <em>example1</em> excerpt of code which you might want to use as a starting point to develop your LK-compliant application.
Please note that the current version <i>is a work-in-progress</i>, which we release to ease discussion and joint work on it.

#How can I get it?
LK is distributed under <a href="https://www.gnu.org/licenses/gpl.html" target="_blank">GPL v3 license</a> + <a href="https://en.wikipedia.org/wiki/GPL_linking_exception">linking exception</a> .
Please download preliminary version 0.2 <a href="https://github.com/HiPeRT/LightKer" target="_blank">here</a>.

#Use LK
- We provide LK as free software, under GPL license with linking exception to make it usable also by proprietary SW developers.
- If you plan to use LK in your project, please add a reference to this website in your documentation
- If you plan to use LK in any scientific publication, please add a reference to the following paper: <i>Paolo Burgio, "Enabling predictable parallelism in single-GPU systems with persistent CUDA threads", in Proceedings of the work-in-progress section of 28th Euromicro Conference on Real-Time Systems (ECRTS16)</i>


# <img src="img/lk-ga-logo.png" style="height:40px">&nbsp;&nbsp;Scheduled Light Kernel</h2>
Scheduled Light Kernel (SLK) is a version of LK used to accelerate  Genetic Algorithms. Please refer to the following paper for more information:
	<br><br>
	<i>Nicola Capodieci and Paolo Burgio, "Efficient Implementation of 
Genetic Algorithms on GP-GPU with Scheduled Persistent CUDA Threads.", 
in  PAAP 2015: 6-12</i>. DOI bookmark: <a href="https://www.computer.org/csdl/proceedings/paap/2015/9117/00/9117a006-abs.html" target="_blank">https://www.computer.org/csdl/proceedings/paap/2015/9117/00/9117a006-abs.html</a>
	
	<h2>Acknowledgements</h2>
	<p>LK is a side research project of the <a href="http://hercules2020.eu/" target="_blank">Hercules Project</a> from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 688860. </p>
	<p>Special thanks to Serena, who implemented the first version of LK</p>

	<h2>Contacts</h2>
	- Paolo Burgio - <a href="mailto:paolo.burgio@unimore.it">paolo.burgio@unimore.it</a><br>
	- (For Scheduled Light Kernel) Nicola Capodieci - <a href="mailto:nicola.capodieci@unimore.it">nicola.capodieci@unimore.it</a><br>
	- Serena Ziviani -  <a href="mailto:ziviani.serena@gmail.com">ziviani.serena@gmail.com</a><br>

	

</body></html>