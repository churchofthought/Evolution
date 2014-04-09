cd %USERPROFILE%\Documents\Dropbox\Evolution
mkdir %TEMP%\cdk
cl cdk\dscale.c cdk\fscale.c cdk\fslider.c cdk\scale.c cdk\slider.c cdk\uscale.c cdk\uslider.c cdk\alphalist.c cdk\binding.c cdk\button.c cdk\buttonbox.c cdk\calendar.c cdk\cdk.c cdk\cdk_compat.c cdk\cdk_display.c cdk\cdk_objs.c cdk\cdk_params.c cdk\cdkscreen.c cdk\debug.c cdk\dialog.c cdk\draw.c cdk\entry.c cdk\fselect.c cdk\get_index.c cdk\get_string.c cdk\graph.c cdk\histogram.c cdk\itemlist.c cdk\label.c cdk\marquee.c cdk\matrix.c cdk\mentry.c cdk\menu.c cdk\popup_dialog.c cdk\popup_label.c cdk\position.c cdk\radio.c cdk\scroll.c cdk\selection.c cdk\swindow.c cdk\select_file.c cdk\template.c cdk\traverse.c cdk\version.c cdk\view_file.c cdk\view_info.c cdk\viewer.c lib/pdcurses.lib Winmm.lib User32.lib Advapi32.lib Shell32.lib Kernel32.lib /Iinclude /Icdk\include /Fo%TEMP%\cdk\ /Ox

del lib\cdk.lib
cd %TEMP%\cdk
lib /out:%USERPROFILE%\Documents\Dropbox\Evolution\lib\cdk.lib dscale.obj fscale.obj fslider.obj scale.obj slider.obj uscale.obj uslider.obj alphalist.obj binding.obj button.obj buttonbox.obj calendar.obj cdk.obj cdk_compat.obj cdk_display.obj cdk_objs.obj cdk_params.obj cdkscreen.obj debug.obj dialog.obj draw.obj entry.obj fselect.obj get_index.obj get_string.obj graph.obj histogram.obj itemlist.obj label.obj marquee.obj matrix.obj mentry.obj menu.obj popup_dialog.obj popup_label.obj position.obj radio.obj scroll.obj selection.obj swindow.obj select_file.obj template.obj traverse.obj version.obj view_file.obj view_info.obj viewer.obj

cd %USERPROFILE%\Documents\Dropbox\Evolution\