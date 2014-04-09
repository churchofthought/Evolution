cd %USERPROFILE%\Documents\Dropbox\Evolution\
del %TEMP%\opencl_evolution.exe
set OPENCL_SDK_ROOT=%INTELOCLSDKROOT%
cl opencl_evolution.c "%OPENCL_SDK_ROOT%lib\x64\OpenCL.lib" lib/cdk.lib lib/pdcurses.lib Winmm.lib User32.lib Advapi32.lib Shell32.lib Kernel32.lib /Iinclude /Iinclude\windows /Icdk\include /I"%OPENCL_SDK_ROOT%include" /Fo%TEMP%\ /Fe%TEMP%\ /Ox
start "OpenCL Evolution" %TEMP%\opencl_evolution