# $Id: Makefile.in,v 1.19 2008/10/30 23:34:32 tom Exp $
#
# Makefile for the examples directory.
#
SHELL		= /bin/sh

prefix		= /usr/local
exec_prefix	= ${prefix}

srcdir		= .


CC		= gcc
RM		= rm -f

LIBS		= -lcdk -lncurses 

LIBTOOL		=  
LIBTOOL_CLEAN	= 
LIBTOOL_LINK	= ${CC}
LINK		= $(LIBTOOL_LINK)

x		= 
o		= .o

CFLAGS		= -g -O2 
CPPFLAGS	= -DHAVE_CONFIG_H -I../include -I$(srcdir)/../include -I.   -D_DARWIN_C_SOURCE
LDFLAGS		= -L.. 

BINS	= \
	alphalist_ex$x \
	bind_ex$x \
	buttonbox_ex$x \
	calendar_ex$x \
	cdkscreen_ex$x \
	dialog_ex$x \
	entry_ex$x \
	fscale_ex$x \
	fselect_ex$x \
	fslider_ex$x \
	graph_ex$x \
	hello_ex$x \
	histogram_ex$x \
	itemlist_ex$x \
	label_ex$x \
	lowerCDKObject_ex$x \
	marquee_ex$x \
	matrix_ex$x \
	mentry_ex$x \
	menu_ex$x \
	position_ex$x \
	preprocess_ex$x \
	radio1_ex$x \
	radio_ex$x \
	raiseCDKObject_ex$x \
	scale_ex$x \
	scroll_ex$x \
	selection_ex$x \
	slider_ex$x \
	subwindow_ex$x \
	swindow_ex$x \
	template_ex$x \
	traverse_ex$x \
	uslider_ex$x \
	viewer_ex$x \
	viewer_ex2$x

LINKIT	= $(LINK) $(CFLAGS) $(CPPFLAGS) $? -o $@ $(LDFLAGS) $(LIBS)

all : $(BINS)

alphalist_ex$x :	alphalist_ex.c		; $(LINKIT)
bind_ex$x :		bind_ex.c		; $(LINKIT)
buttonbox_ex$x :	buttonbox_ex.c		; $(LINKIT)
calendar_ex$x :		calendar_ex.c		; $(LINKIT)
cdkscreen_ex$x :	cdkscreen_ex.c		; $(LINKIT)
dialog_ex$x :		dialog_ex.c		; $(LINKIT)
entry_ex$x :		entry_ex.c		; $(LINKIT)
fscale_ex$x :		fscale_ex.c		; $(LINKIT)
fselect_ex$x :		fselect_ex.c		; $(LINKIT)
fslider_ex$x :		fslider_ex.c		; $(LINKIT)
graph_ex$x :		graph_ex.c		; $(LINKIT)
hello_ex$x :		hello_ex.c		; $(LINKIT)
histogram_ex$x :	histogram_ex.c		; $(LINKIT)
itemlist_ex$x :		itemlist_ex.c		; $(LINKIT)
label_ex$x :		label_ex.c		; $(LINKIT)
lowerCDKObject_ex$x :	lowerCDKObject_ex.c	; $(LINKIT)
marquee_ex$x :		marquee_ex.c		; $(LINKIT)
matrix_ex$x :		matrix_ex.c		; $(LINKIT)
mentry_ex$x :		mentry_ex.c		; $(LINKIT)
menu_ex$x :		menu_ex.c		; $(LINKIT)
position_ex$x :		position_ex.c		; $(LINKIT)
preprocess_ex$x :	preprocess_ex.c		; $(LINKIT)
radio1_ex$x :		radio1_ex.c		; $(LINKIT)
radio_ex$x :		radio_ex.c		; $(LINKIT)
raiseCDKObject_ex$x :	raiseCDKObject_ex.c	; $(LINKIT)
scale_ex$x :		scale_ex.c		; $(LINKIT)
scroll_ex$x :		scroll_ex.c		; $(LINKIT)
selection_ex$x :	selection_ex.c		; $(LINKIT)
slider_ex$x :		slider_ex.c		; $(LINKIT)
subwindow_ex$x :	subwindow_ex.c		; $(LINKIT)
swindow_ex$x :		swindow_ex.c		; $(LINKIT)
template_ex$x :		template_ex.c		; $(LINKIT)
traverse_ex$x :		traverse_ex.c		; $(LINKIT)
uslider_ex$x :		uslider_ex.c		; $(LINKIT)
viewer_ex$x :		viewer_ex.c		; $(LINKIT)
viewer_ex2$x :		viewer_ex2.c		; $(LINKIT)
#
# Standard clean directives.
#
clean::
	-$(LIBTOOL_CLEAN) $(RM) *.o core $(BINS)

distclean:: clean
	$(RM) Makefile
