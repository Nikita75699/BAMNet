use Cwd qw(abs_path);
use File::Basename qw(dirname);

my $here = abs_path(dirname(__FILE__));

$ENV{'BIBINPUTS'} = join(
  ':',
  grep { defined $_ && length $_ }
  (
    $here,
    ($ENV{'BIBINPUTS'} // ''),
  )
);

$ENV{'BSTINPUTS'} = join(
  ':',
  grep { defined $_ && length $_ }
  (
    "$here/bst",
    $here,
    ($ENV{'BSTINPUTS'} // ''),
  )
);
