# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

# enforce using portable C locale
LC_ALL=C
export LC_ALL

action() {
  if (test $mode = 0) then
    rm -f ../../$1
  elif (! cmp -s $1 ../../$1) then
    if (test -z "$2" || test -e ../../$2) then
      cp $1 ../../
      if (test $mode = 2) then
        echo "Updating src/$1"
      fi
    fi
  elif (test $mode = 1) then
    echo "Installing src/$1"
  fi
}

# list all package files that need to be installed

action pair_uma.cpp
action pair_uma.h

# also install Kokkos styles when KOKKOS package is present

if (test -e ../../KOKKOS) then
  if (test $mode = 1) then
    if (test ! -e ../../pair_uma_kokkos.cpp) then
      :
    fi
  fi
fi
