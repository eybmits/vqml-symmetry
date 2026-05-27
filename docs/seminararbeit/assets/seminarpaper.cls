\ProvidesClass{assets/seminarpaper}[Custom class for seminar papers]
\LoadClass{article}
%\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage[nottoc]{tocbibind}
\RequirePackage{fancyhdr}
\RequirePackage{cmap}
\RequirePackage[english,ngerman]{babel}
\RequirePackage{iflang}
\RequirePackage{fancyhdr}
\RequirePackage{geometry}
\RequirePackage{amsmath}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[capitalise,german]{cleveref}
\RequirePackage{subcaption}
\RequirePackage{listings}
\creflabelformat{equation}{#2#1#3}



\selectlanguage{ngerman}

\AddToHook{shipout/firstpage}{%
    \put (0.30\paperwidth,-0.95\paperheight){\begin{tikzpicture}
\node[opacity=.4, anchor=east] {\includegraphics[width=.6\paperwidth]{assets/lmu_seal.pdf}};
\end{tikzpicture}}%
}

\widowpenalty10000
\clubpenalty10000

\renewcommand{\abstract}[1]{\gdef\abstracttext{#1}}
\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand{\seminar}[1]{\gdef\@seminar{#1}}

\newcommand{\quickcharcount}[1]{%
  \immediate\write18{texcount -1 -sum -merge -char #1.tex > #1-chars.sum }%
  \input{#1-chars.sum}%
}

\newcommand{\quickwordcount}[1]{%
  \immediate\write18{texcount -1 -sum -merge #1.tex > #1-words.sum }%
  \input{#1-words.sum}%
}

\newcommand{\printCount}[1]{
    \vspace{1cm}
    \noindent{\textcolor{red}{\Huge Zeichenanzahl: \textbf{\quickcharcount{#1}}}}
}

\newenvironment{customabstract}
  {\small \begin{center}
  \bfseries Zusammenfassung
  \end{center}
  \begin{quote}}
  {\end{quote}}

\renewcommand{\maketitle}{
    \pagenumbering{Roman}
    \newcommand{\mytitle}{\@title}
    \begin{titlepage}
    	\centering
            \vspace*{-2.5cm}
            \noindent\makebox[\linewidth]{
                \includegraphics[width=.9\paperwidth]{assets/lmu_header.pdf}
            }\\
            \vspace*{1.75cm}
            \centering
            {\scshape\large \@seminar}\\
            \vspace*{0.40cm}
            {\huge\bfseries\@title}\\
            \vspace*{1.75cm}
    	\large \begin{tabular}{ l l }
    	  \textsc{Bearbeiter}:&{\@author}\\
    	  \\\\
    	  \textsc{Betreuer:}&{\@supervisor}\\
    	  \\
    	  \textsc{Aufgabensteller:}&{Prof. Dr. Claudia Linnhoff-Popien}\\
    	\end{tabular}
    	\par
    	\vspace*{1.75cm}
    	\begin{otherlanguage}{ngerman}
    		{\large Dokument erstellt\par\today\par}
    	\end{otherlanguage}
    	\newpage
    	\thispagestyle{empty}
    	\mbox{}
    	\newpage
            \cleardoublepage
            \begin{center}
            	\LARGE
            	\mytitle
            \end{center}
            \vspace*{.75cm}
            \begin{customabstract}
            \abstracttext
            \end{customabstract}

            \vspace*{.75cm}
            \tableofcontents
            \newpage
            \thispagestyle{empty}
            \mbox{}
            \newpage
            \cleardoublepage
    \end{titlepage}
    \pagenumbering{arabic}
}