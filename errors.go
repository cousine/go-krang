package krang

import (
	"fmt"
)

type IncompleteInputParameters struct {
	expected int
	given    int
}

func (krangError IncompleteInputParameters) Error() string {
	return fmt.Sprintf("Krang Error: Incomplete input parameters supplied, %d instead of %d", krangError.given, krangError.expected)
}
