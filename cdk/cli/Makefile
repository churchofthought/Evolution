# $Id: Makefile.in,v 1.5 2008/10/30 23:34:49 tom Exp $
#
# Makefile for the cli directory.
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
	cdkalphalist$x \
	cdkcalendar$x \
	cdkdialog$x \
	cdkentry$x \
	cdkfselect$x \
	cdkitemlist$x \
	cdklabel$x \
	cdkmatrix$x \
	cdkmentry$x \
	cdkradio$x \
	cdkscale$x \
	cdkscroll$x \
	cdkselection$x \
	cdkslider$x \
	cdktemplate$x \
	cdkviewer$x

LINKIT	= $(LINK) $(CFLAGS) $(CPPFLAGS) $? -o $@ $(LDFLAGS) $(LIBS)

all : $(BINS)

cdkalphalist$x :	cdkalphalist.c	; $(LINKIT)
cdkcalendar$x :		cdkcalendar.c	; $(LINKIT)
cdkdialog$x :		cdkdialog.c	; $(LINKIT)
cdkentry$x :		cdkentry.c	; $(LINKIT)
cdkfselect$x :		cdkfselect.c	; $(LINKIT)
cdkitemlist$x :		cdkitemlist.c	; $(LINKIT)
cdklabel$x :		cdklabel.c	; $(LINKIT)
cdkmatrix$x :		cdkmatrix.c	; $(LINKIT)
cdkmentry$x :		cdkmentry.c	; $(LINKIT)
cdkradio$x :		cdkradio.c	; $(LINKIT)
cdkscale$x :		cdkscale.c	; $(LINKIT)
cdkscroll$x :		cdkscroll.c	; $(LINKIT)
cdkselection$x :	cdkselection.c	; $(LINKIT)
cdkslider$x :		cdkslider.c	; $(LINKIT)
cdktemplate$x :		cdktemplate.c	; $(LINKIT)
cdkviewer$x :		cdkviewer.c	; $(LINKIT)
#
# Standard clean directives.
#
clean::
	- $(LIBTOOL_CLEAN) $(RM) *.o core $(BINS)

distclean:: clean
	$(RM) Makefile
